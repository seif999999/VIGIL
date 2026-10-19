Rendered simulation poster only (not a live human).

Use a texture SCRFD can detect on the flat poster file:

  copy "..\..\face_detection\my_face.jpeg" face_target.jpg

Do NOT use hmac\test_images\original_1.jpg — SCRFD returns NO_FACE on that file.

Verify offline:

  cd ..\..
  python webots\diagnose_poster_texture.py

World reference: ../textures/face_target.jpg
