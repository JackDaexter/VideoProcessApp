"""
app/services/openshots_service.py — Core OpenShorts processing engine.

This is a faithful adaptation of the OpenShorts (mutonby/openshorts) pipeline,
migrated to our FastAPI backend with GCS storage and Supabase job tracking.

Key technologies (matching the real OpenShorts stack):
  - Transcription    → faster-whisper (word-level timestamps)
  - Clip Selection   → Google Gemini LLM (same Gemini prompt as OpenShorts)
  - Scene Detection  → PySceneDetect ContentDetector
  - Face Tracking    → MediaPipe BlazeFace + SmoothedCameraman
  - Person Fallback  → YOLOv8n
  - 9:16 Reframe     → OpenCV frame-by-frame crop + FFmpeg audio merge
  - Captions         → faster-whisper word timestamps → SRT → FFmpeg burn-in
  - Rendering        → FFmpeg subprocess calls
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import structlog
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from ultralytics import YOLO

from app.config import get_settings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

log = structlog.get_logger(__name__)

# ── Constants (from OpenShorts) ───────────────────────────────────────────────

ASPECT_RATIO = 9 / 16  # Vertical 9:16 for Shorts/Reels/TikTok

# YOLO frame intervals per quality level (None = never run YOLO)
YOLO_FRAME_INTERVALS: dict = {
    "LOW": None,      # 0%  — MediaPipe only, no YOLO fallback
    "MEDIUM": 50,     # ~2% — run YOLO every 50 frames
    "HIGH": 10,       # ~10% — run YOLO every 10 frames
    "PREMIUM": 1,     # 100% — run YOLO on every frame (original behavior)
}

# Gemini prompt — identical to OpenShorts' GEMINI_PROMPT_TEMPLATE
GEMINI_CLIP_PROMPT = """\
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps \
to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. \
Each clip must be between {min_duration} and {max_duration} seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video.
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between {min_duration} and {max_duration} s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

USER PROMPT (use this to bias clip selection):
{user_prompt}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < {min_duration} s or > {max_duration} s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok>",
      "video_description_for_instagram": "<description for Instagram>",
      "video_title_for_youtube_short": "<title for YouTube Short, 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay, max 10 words>"
    }}
  ]
}}
"""

GEMINI_YOUTUBE_PROMPT = """\
You are a YouTube SEO expert. Based on the transcript below, generate:
1. A compelling, SEO-optimized title (max 100 characters)
2. A detailed description with chapter timestamps (max 5000 characters)
3. 15 relevant tags
4. A punchy hook sentence for the thumbnail text

USER GUIDANCE: {user_prompt}

TRANSCRIPT:
{transcript_text}

OUTPUT — RETURN ONLY VALID JSON (no markdown):
{{
  "title": "<YouTube title>",
  "description": "<full YouTube description with chapters>",
  "tags": ["tag1", "tag2", ...],
  "thumbnail_text": "<short hook text for thumbnail overlay>"
}}
"""


# ── Model Singletons ──────────────────────────────────────────────────────────

_whisper_model: Optional[WhisperModel] = None
_yolo_model: Optional[Any] = None
_mp_face_detection: Optional[Any] = None


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        settings = get_settings()
        log.info("whisper_loading", model=settings.whisper_model, device=settings.whisper_device)
        _whisper_model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        log.info("whisper_loaded")
    return _whisper_model


def get_yolo_model() -> Any:
    global _yolo_model
    if _yolo_model is None:
        # Load from the absolute path baked into the Docker image at build time.
        # YOLO_MODEL_PATH is set in the Dockerfile to /app/models/yolov8n.pt,
        # so ultralytics loads the file directly — no download is ever attempted.
        model_path = os.environ.get("YOLO_MODEL_PATH", "/app/models/yolov8n.pt")
        log.info("yolo_loading", path=model_path)
        _yolo_model = YOLO(model_path)
        log.info("yolo_loaded")
    return _yolo_model


def get_face_detection():
    global _mp_face_detection
    if _mp_face_detection is None:
        import mediapipe as mp
        mp_fd = mp.solutions.face_detection
        _mp_face_detection = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return _mp_face_detection


# ── SmoothedCameraman (from OpenShorts) ───────────────────────────────────────

class SmoothedCameraman:
    """
    Handles smooth 9:16 camera reframing for face tracking.
    Ported directly from mutonby/openshorts main.py.
    """

    def __init__(self, output_width: int, output_height: int, video_width: int, video_height: int):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height

        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
            self.crop_width = video_width
            self.crop_height = int(self.crop_width / ASPECT_RATIO)

        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box: Optional[List[int]]) -> None:
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2

    def get_crop_box(self, force_snap: bool = False) -> Tuple[int, int, int, int]:
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.safe_zone_radius:
                direction = 1 if diff > 0 else -1
                speed = 15.0 if abs(diff) > self.crop_width * 0.5 else 3.0
                self.current_center_x += direction * speed
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x

        half_crop = self.crop_width / 2
        self.current_center_x = max(half_crop, min(self.video_width - half_crop, self.current_center_x))

        x1 = max(0, int(self.current_center_x - half_crop))
        x2 = min(self.video_width, int(self.current_center_x + half_crop))
        return x1, 0, x2, self.video_height


class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid face switching.
    Ported from mutonby/openshorts main.py.
    """

    def __init__(self, stabilization_frames: int = 15, cooldown_frames: int = 30):
        self.active_speaker_id: Optional[int] = None
        self.speaker_scores: Dict[int, float] = {}
        self.last_switch_frame = -1000
        self.locked_counter = 0
        self.switch_cooldown = cooldown_frames
        self.next_id = 0
        self.known_faces: List[Dict] = []

    def get_target(self, face_candidates: List[Dict], frame_number: int, width: int) -> Optional[List[int]]:
        current_candidates = []
        for face in face_candidates:
            x, y, w, h = face["box"]
            center_x = x + w / 2
            best_match_id = -1
            min_dist = width * 0.15

            for kf in self.known_faces:
                if frame_number - kf["last_frame"] > 30:
                    continue
                dist = abs(center_x - kf["center"])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf["id"]

            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1

            self.known_faces = [kf for kf in self.known_faces if kf["id"] != best_match_id]
            self.known_faces.append({"id": best_match_id, "center": center_x, "last_frame": frame_number})
            current_candidates.append({"id": best_match_id, "box": face["box"], "score": face["score"]})

        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand["id"]
            raw_score = cand["score"] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        if not current_candidates:
            return None

        best_candidate = None
        max_score = -1
        for cand in current_candidates:
            pid = cand["id"]
            total_score = self.speaker_scores.get(pid, 0)
            if pid == self.active_speaker_id:
                total_score *= 3.0
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate:
            target_id = best_candidate["id"]
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate["box"]
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c["id"] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand["box"]
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate["box"]

        return None


# ── Face / Person Detection ───────────────────────────────────────────────────

def detect_face_candidates(frame: np.ndarray) -> List[Dict]:
    """Detect all faces in a frame using MediaPipe BlazeFace."""
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = get_face_detection().process(rgb_frame)

    candidates = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            candidates.append({"box": [x, y, w, h], "score": w * h})
    return candidates


def detect_person_yolo(frame: np.ndarray) -> Optional[List[int]]:
    """Fallback: detect largest person with YOLOv8 when face detection fails."""
    results = get_yolo_model()(frame, verbose=False, classes=[0])
    best_box = None
    max_area = 0
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = [x1, y1, x2 - x1, y2 - y1]
    return best_box


# ── Transcription (faster-whisper) ───────────────────────────────────────────

async def transcribe_video(video_path: str) -> Dict[str, Any]:
    """
    Transcribe a video using faster-whisper with word-level timestamps.

    Returns:
        {
          "text": "full transcript",
          "segments": [...],
          "words": [{"w": word, "s": start, "e": end}, ...]   ← for Gemini prompt
        }
    """
    log.info("transcribe_start", path=video_path)

    def _run() -> Dict[str, Any]:
        model = get_whisper_model()
        try:
            segments, info = model.transcribe(
                video_path,
                language="en",
                word_timestamps=True,
                beam_size=get_settings().whisper_beam_size,
            )
            full_text = ""
            all_segments = []
            all_words = []
            total_duration = info.duration or 0.0
            last_logged_pct = 0

            for seg in segments:
                full_text += seg.text
                seg_dict = {"start": seg.start, "end": seg.end, "text": seg.text}
                all_segments.append(seg_dict)
                if seg.words:
                    for w in seg.words:
                        all_words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})

                if total_duration > 0:
                    pct = int(seg.end / total_duration * 100)
                    if pct >= last_logged_pct + 10:
                        last_logged_pct = (pct // 10) * 10
                        log.info("transcribe_progress", pct=last_logged_pct)

            return {"text": full_text.strip(), "segments": all_segments, "words": all_words}
        except IndexError as exc:
            # faster-whisper raises "IndexError: tuple index out of range" via PyAV
            # if the video has no audio track or the audio track is unreadable.
            log.error("transcribe_failed_no_audio", path=video_path, error=str(exc))
            raise ValueError(
                "The video doesn't have an audio track. Please verify that the input video has sound."
            ) from exc

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _run)
        log.info("transcribe_done", segments=len(result["segments"]), words=len(result["words"]))
        return result
    except ValueError as exc:
        raise exc
    except Exception as exc:
        log.error("transcribe_failed_unexpected", path=video_path, error=str(exc))
        raise exc


# ── Clip Selection via Gemini LLM ─────────────────────────────────────────────

async def select_clips_with_gemini(
    transcript_text: str,
    words: List[Dict],
    video_duration: float,
    user_prompt: str,
    max_clips: int = 10,
    min_duration: int = 15,
    max_duration: int = 60,
) -> List[Dict[str, Any]]:
    """
    Use Google Gemini (same approach as OpenShorts) to select viral clip moments.

    Returns list of clip dicts:
      [{start, end, video_description_for_tiktok, video_description_for_instagram,
        video_title_for_youtube_short, viral_hook_text}]
    """
    settings = get_settings()
    log.info("gemini_clip_selection_start", duration=video_duration)

    words_json = json.dumps(words[:500], ensure_ascii=False)  # Limit tokens

    prompt = GEMINI_CLIP_PROMPT.format(
        video_duration=round(video_duration, 3),
        transcript_text=transcript_text[:8000],
        words_json=words_json,
        user_prompt=user_prompt,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    def _call_gemini() -> List[Dict]:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        clips = parsed.get("shorts", [])

        # Enforce duration constraints
        valid = []
        for c in clips:
            duration = c["end"] - c["start"]
            if 0 <= c["start"] < c["end"] <= video_duration and min_duration <= duration <= max_duration:
                valid.append(c)
        return valid[:max_clips]

    clips = await asyncio.get_event_loop().run_in_executor(None, _call_gemini)
    log.info("gemini_clip_selection_done", clips=len(clips))
    return clips


# ── Scene Detection ───────────────────────────────────────────────────────────

async def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect scene boundaries using PySceneDetect."""
    log.info("scene_detect_start", path=video_path)

    def _run() -> List[Tuple[float, float]]:
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(video)
        return [(s.get_seconds(), e.get_seconds()) for s, e in manager.get_scene_list()]

    scenes = await asyncio.get_event_loop().run_in_executor(None, _run)
    log.info("scene_detect_done", scenes=len(scenes))
    return scenes


# ── Video Duration ────────────────────────────────────────────────────────────

async def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using OpenCV."""
    def _run() -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0.0

    return await asyncio.get_event_loop().run_in_executor(None, _run)


# ── FFmpeg Helpers ────────────────────────────────────────────────────────────

def _ffmpeg(args: List[str]) -> None:
    """Run an FFmpeg command, raising on error."""
    cmd = ["ffmpeg", "-y", "-threads", "0"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")


async def extract_audio(video_path: str, audio_path: str) -> str:
    """Extract audio track to a WAV file — much faster for whisper than reading the full video."""
    def _run() -> None:
        _ffmpeg([
            "-i", video_path,
            "-vn",              # drop video stream
            "-acodec", "pcm_s16le",
            "-ar", "16000",     # 16kHz — whisper's native sample rate
            "-ac", "1",         # mono
            audio_path,
        ])
    await asyncio.get_event_loop().run_in_executor(None, _run)
    return audio_path


async def cut_clip(source_path: str, output_path: str, start: float, end: float) -> str:
    """Cut a segment from a video using FFmpeg stream copy (fast, no re-encode)."""
    log.info("cut_clip", start=start, end=end)
    duration = end - start
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        _ffmpeg([
            "-ss", str(start),
            "-t", str(duration),
            "-i", source_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ])

    await asyncio.get_event_loop().run_in_executor(None, _run)
    return output_path


async def extract_thumbnail(video_path: str, output_path: str, timestamp: float = 5.0) -> str:
    """Extract a single JPEG frame at the given timestamp."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        _ffmpeg([
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            output_path,
        ])

    await asyncio.get_event_loop().run_in_executor(None, _run)
    return output_path


# ── 9:16 Reframe — TRACK mode (MediaPipe + SmoothedCameraman) ─────────────────

async def reframe_to_vertical_track(
    source_path: str,
    output_path: str,
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """
    Reframe video to 9:16 using MediaPipe face tracking + SmoothedCameraman.
    This is the TRACK mode from OpenShorts: it follows the speaker's face smoothly.

    Process:
      1. Read frames with OpenCV
      2. Detect faces via MediaPipe (YOLO fallback)
      3. SmoothedCameraman decides crop box
      4. Write cropped frames to temp silent video
      5. Merge original audio with FFmpeg
    """
    log.info("reframe_track_start", src=source_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_silent = output_path.replace(".mp4", "_silent.mp4")

    def _run() -> None:
        settings = get_settings()
        quality = settings.reframe_quality.upper()
        yolo_interval = YOLO_FRAME_INTERVALS.get(quality, 1)
        log.info("reframe_track_quality", quality=quality, yolo_interval=yolo_interval)

        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cameraman = SmoothedCameraman(output_width, output_height, v_width, v_height)
        tracker = SpeakerTracker()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_silent, fourcc, fps, (output_width, output_height))

        last_yolo_box: Optional[List[int]] = None
        last_yolo_frame: int = -1000

        for frame_num in tqdm(range(total_frames), desc="Reframing", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            face_candidates = detect_face_candidates(frame)
            face_box = tracker.get_target(face_candidates, frame_num, v_width)

            # YOLO fallback — frequency controlled by REFRAME_QUALITY
            if face_box is None:
                if yolo_interval is None:
                    # LOW: never run YOLO, reuse last known position
                    face_box = last_yolo_box
                elif frame_num - last_yolo_frame >= yolo_interval:
                    # Run YOLO and cache the result
                    face_box = detect_person_yolo(frame)
                    last_yolo_box = face_box
                    last_yolo_frame = frame_num
                else:
                    # Reuse cached YOLO result from N frames ago
                    face_box = last_yolo_box

            cameraman.update_target(face_box)
            x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=(frame_num == 0))

            # Crop
            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[1] == 0 or cropped.shape[0] == 0:
                cropped = frame

            # Resize to output dimensions
            resized = cv2.resize(cropped, (output_width, output_height))
            out.write(resized)

        cap.release()
        out.release()

        # Merge audio from source
        _ffmpeg([
            "-i", tmp_silent,
            "-i", source_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-crf", "23",
            "-preset", "fast",
            output_path,
        ])

        # Cleanup silent temp
        try:
            os.remove(tmp_silent)
        except FileNotFoundError:
            pass

    await asyncio.get_event_loop().run_in_executor(None, _run)
    log.info("reframe_track_done", output=output_path)
    return output_path


async def reframe_to_vertical_general(
    source_path: str,
    output_path: str,
    output_width: int = 1080,
    output_height: int = 1920,
) -> str:
    """
    GENERAL mode: blurred background fill for 9:16.
    Scales the video to fit height, pads sides with a blurred version.
    """
    log.info("reframe_general_start", src=source_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        vf = (
            f"split[original][copy];"
            f"[copy]scale={output_width}:{output_height}:force_original_aspect_ratio=increase,"
            f"crop={output_width}:{output_height},boxblur=20:20[bg];"
            f"[original]scale=-1:{output_height}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        _ffmpeg([
            "-i", source_path,
            "-vf", vf,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-crf", "23",
            "-preset", "ultrafast",
            output_path,
        ])

    await asyncio.get_event_loop().run_in_executor(None, _run)
    log.info("reframe_general_done", output=output_path)
    return output_path


async def reframe_to_vertical(source_path: str, output_path: str) -> str:
    """Dispatch to TRACK or GENERAL crop mode based on settings."""
    settings = get_settings()
    if settings.crop_mode == "TRACK":
        return await reframe_to_vertical_track(source_path, output_path)
    return await reframe_to_vertical_general(source_path, output_path)


# ── SRT Caption Generation ────────────────────────────────────────────────────

def segments_to_srt(segments: List[Dict], offset: float = 0.0) -> str:
    """Convert faster-whisper segments to SRT format, adjusting timestamps by offset."""

    def _fmt(secs: float) -> str:
        secs = max(0.0, secs - offset)
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int((secs % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines += [str(i), f"{_fmt(seg['start'])} --> {_fmt(seg['end'])}", seg["text"].strip(), ""]
    return "\n".join(lines)


async def burn_captions(source_path: str, srt_path: str, output_path: str) -> str:
    """Burn SRT subtitles into video — styled for vertical Shorts (large centered text)."""
    log.info("burn_captions_start")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    style = (
        "FontName=Arial,"
        "FontSize=20,"
        "PrimaryColour=&HFFFFFF,"
        "OutlineColour=&H000000,"
        "Outline=2,"
        "Alignment=2,"
        "MarginV=80"
    )

    def _run() -> None:
        _ffmpeg([
            "-i", source_path,
            "-vf", f"subtitles={srt_path}:force_style='{style}'",
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-crf", "23",
            "-preset", "fast",
            output_path,
        ])

    await asyncio.get_event_loop().run_in_executor(None, _run)
    return output_path


# ── YouTube Metadata via Gemini ───────────────────────────────────────────────

async def generate_youtube_metadata_with_gemini(
    transcript_text: str,
    user_prompt: str,
    channel_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Gemini to generate YouTube title, description, tags, and thumbnail text.
    """
    settings = get_settings()
    log.info("gemini_youtube_metadata_start")

    ctx = f"\nChannel context: {channel_context}" if channel_context else ""
    prompt = GEMINI_YOUTUBE_PROMPT.format(
        user_prompt=user_prompt + ctx,
        transcript_text=transcript_text[:8000],
    )

    def _call_gemini() -> Dict[str, Any]:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)

    metadata = await asyncio.get_event_loop().run_in_executor(None, _call_gemini)
    log.info("gemini_youtube_metadata_done", title=metadata.get("title", "")[:50])
    return metadata
