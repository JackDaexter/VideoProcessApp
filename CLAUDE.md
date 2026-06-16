# App Server description

You are a Server that handle video processing like clip generation, AI shorts, Youtube studio. You will use (Openshorts)[https://github.com/mutonby/openshorts] as a reference to build this server.
We will make api call to this server sending video and get the result.

There will be endpoint for :
- Clip Generator
- AI Shorts
- Youtube studio


# API endpoints 

You should create api endpoint for the following operations :
- Clip Generator
- AI Shorts
- Youtube studio

# Clip Generator API endpoint

This endpoint will be used to generate clip from a video.

The request will be a POST request to /api/clip-generator with the following parameters :
- video: The video to process
- prompt: The prompt to use for clip generation

The response will be a JSON object with the following parameters :
- status: The status of the request
- result: The result of the request

# AI Shorts API endpoint

This endpoint will be used to generate AI shorts from a video.

The request will be a POST request to /api/ai-shorts with the following parameters :
- video: The video to process
- prompt: The prompt to use for AI shorts generation

The response will be a JSON object with the following parameters :
- status: The status of the request
- result: The result of the request

# Youtube Studio API endpoint

This endpoint will be used to generate youtube studio from a video.

The request will be a POST request to /api/youtube-studio with the following parameters :
- video: The video to process
- prompt: The prompt to use for youtube studio generation

The response will be a JSON object with the following parameters :
- status: The status of the request
- result: The result of the request