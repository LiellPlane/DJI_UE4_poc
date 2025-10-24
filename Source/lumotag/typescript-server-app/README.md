
test server with

curl -X GET http://LIELLOMEN.broadband:8080/api/v1/gamestate -H "X-User-ID: my_username"



test looping script:



#!/bin/bash
while true; do
  echo "$(date): Testing server..."
  if curl -X GET http://LIELLOMEN.broadband:8080/api/v1/gamestate -H "X-User-ID: my_username" --connect-timeout 5 --max-time 10 -s; then
    echo " ✅ Success"
  else
    echo " ❌ Failed (exit code: $?)"
  fi
  echo "---"
  sleep 2
done


or


while true; do echo "$(date): Testing server..."; if curl -X GET http://LIELLOMEN.broadband:8080/api/v1/gamestate -H "x-device-id: 12345" --connect-timeout 5 --max-time 10 -s; then echo " ✅ Success"; else echo " ❌ Failed"; fi; echo "---"; sleep 2; done

WINDOWS user tagged curl:
curl -X POST http://LIELLOMEN.broadband:8080/api/v1/events -H "Content-Type: application/json" -H "x-device-id: bc1ad358bb" -d "{\"event_type\": \"PlayerTagged\", \"tag_id\": \"1\", \"image_ids\": [\"img_001_1727033058\", \"img_002_1727033059\"]}"

curl -X GET "http://LIELLOMEN.broadband:8080/api/v1/gamestate" -H "x-device-id: bc1ad358bb" -H "Content-Type: application/json" -v

curl -X GET http://LIELLOMEN.broadband:8080/api/v1/gamestate -H "x-device-id: bc1ad358bb"

run with pnpm build:start:dashboard (or remove dashboard) - then go to localhost:3001 to get the dash