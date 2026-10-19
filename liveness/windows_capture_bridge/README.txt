Optional C# bridge scaffold (not used in the working path).

VIGIL uses Python WinRT bindings (windows_camera_capture.py) with
MediaCapture.create_frame_reader_async because MediaFrameReader static
factories are not exposed in the Python/C# projections tested here.

Build (may fail on some SDK versions):
  dotnet build -c Release

Primary capture test:
  python ..\windows_camera_capture.py --device-name "Nazmy (Windows Virtual Camera)" --frames 30
