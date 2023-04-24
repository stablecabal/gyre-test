#!/bin/bash

# SIMPLE: Test list engines
# curl -v localhost:5000/v1/engines/list

# SIMPLE: Test upscale
curl -v localhost:5000/v1/generation/hat-gan-x4/image-to-image/upscale \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -H 'Accept: image/png' \
  -F 'image=@"man.png"' \
  -F "height=1000" \
  -o test_out.png

# SIMPLE: Test generate with form-encoded
curl -v localhost:5000/v1/generation/stable-diffusion-v1-5/text-to-image \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -H 'Accept: image/png' \
  -F 'text_prompts[0][text]=A%20Teddybear' \
  -o test_out.png

# ADVANCED: Test basic list engines gateway
# curl -v -X GET localhost:5000/grpcgateway/engines

# ADVANCED: And POST
# curl -v -X POST --data '{"task_group": "GENERATE"}' -H "Content-type: application/json" localhost:5000/grpcgateway/engines

# ADVANCED: Test bad content type
# curl -v -X POST -H "Content-type: image/png" localhost:5000/grpcgateway/engines

# ADVANCED: Test direct generation
# curl -X POST "http://localhost:5000/grpcgateway/generate" \
#   -H 'Content-Type: application/json' \
#   --data-raw '{
#     "engineId": "stable-diffusion-v1-5",
#     "requestId": "f3de102a-7142-4a73-8116-d23e4d37525a",
#     "prompt": [
#         {
#         "text": "A teddybear"
#         }
#     ]
#   }' \
#   -o test_out.png

# ADVANCED: Test async generation
# curl -X POST "http://localhost:5000/grpcgateway/asyncGenerate" \
#   -H 'Content-Type: application/json' \
#   --data-raw '{
#     "engineId": "stable-diffusion-v1-5",
#     "requestId": "f3de102a-7142-4a73-8116-d23e4d37525a",
#     "prompt": [
#         {
#         "text": "A teddybear"
#         }
#     ]
#   }' \
#   -o test_asynchandle.json

# sleep 30s

# curl -X POST "http://localhost:5000/grpcgateway/asyncResult" \
#   -H 'Content-Type: application/json' \
#   --data @test_asynchandle.json \
#   -o test_out.json


