# WildLifeAI

Windows Forms app that watches RTSP camera streams, runs YOLO object detection (Emgu.CV / DNN),
saves detections, and optionally uploads photos over FTP.

## Configuration

No credentials or addresses are stored in the repository. Provide settings as environment
variables, or in `appsettings.local.json` next to the executable (copy
`appsettings.local.example.json`; the file is git-ignored).

| Key | Purpose |
|---|---|
| `CAMERA_URL_1` ... `CAMERA_URL_N` | RTSP URLs, including the camera username/password (required: at least `CAMERA_URL_1`) |
| `CAMERA_NAME_1` ... `CAMERA_NAME_N` | Display names (optional; default `Camera N`) |
| `FTP_HOST`, `FTP_USERNAME`, `FTP_PASSWORD`, `FTP_PATH` | FTP upload target (optional; uploads are skipped if unset) |

Use unique, strong passwords for the cameras and FTP account.

Model files (`*.onnx`, `*.weights`, `*.cfg`) are not included; see `.gitignore`.
