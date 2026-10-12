using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace WildLifeAI
{
    public partial class Form1 : Form
    {
        // Core detection fields
        private Net _net;
        private List<string> _labels;
        private VideoCapture _capture;
        private readonly object _captureLock = new object();
        private bool _running = false;
        private bool _resized = false;
        private bool _detectionEnabled = true;

        // Performance optimization
        private readonly object _frameLock = new object();
        private Mat _latestFrame;
        private Task _detectionTask;
        private CancellationTokenSource _cancellationTokenSource;
        private int _frameSkipCounter = 0;
        private const int FRAME_SKIP = 15;  // Reduced from 30 for better small object tracking

        private TrackBar trackBar1;
        private PictureBox pictureBox1;

        // ============ Multi-camera support ============
        private volatile int _currentCameraIndex = 0;
        private System.Windows.Forms.Timer _cameraSwitchTimer;
        private readonly int CAMERA_SWITCH_INTERVAL_MS = 30000;
        private volatile bool _isSwitchingCamera = false;
        private Label _cameraIndicatorLabel;

        // Camera RTSP URLs (which contain credentials) and display names come from configuration:
        // CAMERA_URL_1..N and CAMERA_NAME_1..N. See README.md.
        private readonly string[] _cameraUrls = LoadCameraUrls();
        private readonly string[] _cameraNames = LoadCameraNames();

        private static string[] LoadCameraUrls()
        {
            var urls = AppConfig.GetIndexed("CAMERA_URL");
            if (urls.Length == 0)
                throw new InvalidOperationException("Configuration value 'CAMERA_URL_1' is not set. Set CAMERA_URL_1..N as environment variables or in appsettings.local.json (see README.md).");
            return urls;
        }

        private static string[] LoadCameraNames()
        {
            int count = AppConfig.GetIndexed("CAMERA_URL").Length;
            var names = new string[count];
            for (int i = 0; i < count; i++)
                names[i] = AppConfig.Get($"CAMERA_NAME_{i + 1}") ?? $"Camera {i + 1}";
            return names;
        }
        // ===================================================

        // Location tracking for duplicate prevention
        private Dictionary<string, DateTime> _lastSaveTime = new Dictionary<string, DateTime>();
        private Dictionary<string, List<TrackedObject>> _trackedObjects = new Dictionary<string, List<TrackedObject>>();
        private readonly int SAVE_INTERVAL_SECONDS = 30;
        private readonly int LOCATION_TOLERANCE = 100;
        private readonly int TRACKING_TIMEOUT_MINUTES = 1440;

        // File handling
        private readonly string _photoDirectory = @"D:\AIPhotos";

        // FTP upload configuration
        private Queue<string> _uploadQueue = new Queue<string>();
        private Task _uploadTask;
        private HashSet<string> _uploadedFiles = new HashSet<string>();
        private readonly string _ftpHost = AppConfig.Get("FTP_HOST") ?? string.Empty;
        private string _ftpUsername = AppConfig.Get("FTP_USERNAME") ?? string.Empty;
        private string _ftpPassword = AppConfig.Get("FTP_PASSWORD") ?? string.Empty;
        private readonly string _ftpPath = AppConfig.Get("FTP_PATH") ?? "/";

        // Detection results queue
        private ConcurrentQueue<DetectionResult> _detectionResults = new ConcurrentQueue<DetectionResult>();

        // Resolution settings
        private readonly Size CAMERA_RESOLUTION = new Size(3840, 2160);
        private readonly Size DISPLAY_RESOLUTION = new Size(1280, 720);
        private readonly int JPEG_QUALITY = 100;

        // YOLOv11 specific settings
        private readonly int YOLO_INPUT_SIZE = 640;
        private readonly int YOLO_INPUT_WIDTH = 640;
        private readonly int YOLO_INPUT_HEIGHT = 640;
        public float CONFIDENCE_THRESHOLD = 0.30f;  // Lowered from 0.40f for better small object detection
        private readonly float NMS_THRESHOLD = 0.5f;
        private TextBox tb1;

        // ============ TILED DETECTION SETTINGS ============
        private readonly bool ENABLE_TILED_DETECTION = true;
        private readonly int TILE_SIZE = 640;
        private readonly int TILE_OVERLAP = 128;  // Overlap to catch objects at tile boundaries
        private readonly float SMALL_OBJECT_CONFIDENCE_MULTIPLIER = 0.7f;  // Lower threshold for small objects

        // Cached ONNX Session for performance
        private InferenceSession _onnxSession;
        private readonly object _sessionLock = new object();
        // ==================================================

        public Form1()
        {
            InitializeComponent();
            Directory.CreateDirectory(_photoDirectory);
        }

        private class DetectionResult
        {
            public Rectangle Rect { get; set; }
            public string Label { get; set; }
            public float Confidence { get; set; }
            public Mat FrameSnapshot { get; set; }
            public int CameraIndex { get; set; }
        }

        private class TrackedObject
        {
            public Rectangle Location { get; set; }
            public DateTime FirstSeen { get; set; }
            public DateTime LastSeen { get; set; }
            public string UniqueId { get; set; }

            public bool IsNearLocation(Rectangle rect, int tolerance)
            {
                int centerX1 = Location.X + Location.Width / 2;
                int centerY1 = Location.Y + Location.Height / 2;
                int centerX2 = rect.X + rect.Width / 2;
                int centerY2 = rect.Y + rect.Height / 2;

                double distance = Math.Sqrt(Math.Pow(centerX2 - centerX1, 2) + Math.Pow(centerY2 - centerY1, 2));
                return distance <= tolerance;
            }
        }

        private void LoadYolo()
        {
            try
            {
                string modelPath = "yolov8x-oiv7.onnx";
                string labelsFile = "oiv7_classes.names";

                if (!File.Exists(modelPath))
                {
                    MessageBox.Show($"Model file not found: {modelPath}\n\nPlease place yolo11x.onnx in the application directory.");
                    return;
                }

                if (!File.Exists(labelsFile))
                {
                    MessageBox.Show($"Labels file not found: {labelsFile}\n\nPlease place coco.names in the application directory.");
                    return;
                }

                rtb1.AppendText("Loading YOLOv11x model...\n");
                _net = DnnInvoke.ReadNetFromONNX(modelPath);

                try
                {
                    CvInvoke.UseOpenCL = true;
                    _net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                    _net.SetPreferableTarget(Emgu.CV.Dnn.Target.OpenCL);

                    if (CvInvoke.HaveOpenCL && CvInvoke.UseOpenCL)
                    {
                        rtb1.AppendText("✓ Using GPU acceleration\n");
                    }
                    else
                    {
                        throw new Exception("OpenCL not available");
                    }
                }
                catch
                {
                    _net.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
                    _net.SetPreferableTarget(Emgu.CV.Dnn.Target.Cpu);
                    rtb1.AppendText("⚠ Using CPU\n");
                }

                _labels = File.ReadAllLines(labelsFile).ToList();

                // Initialize cached ONNX session for tiled detection
                InitializeOnnxSession(modelPath);

                rtb1.AppendText($"✓ Model loaded - {_labels.Count} classes\n");
                rtb1.AppendText($"✓ Tiled detection: {(ENABLE_TILED_DETECTION ? "ENABLED" : "DISABLED")}\n");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading YOLOv11: {ex.Message}");
                rtb1.AppendText($"ERROR: {ex.Message}\n");
            }
        }

        // Initialize ONNX session once for better performance
        private void InitializeOnnxSession(string modelPath)
        {
            try
            {
                var sessionOptions = new SessionOptions();

                // Try to use GPU if available
                try
                {
                    sessionOptions.AppendExecutionProvider_DML(0);  // DirectML for Windows GPU
                    rtb1.AppendText("✓ ONNX using DirectML GPU\n");
                }
                catch
                {
                    try
                    {
                        sessionOptions.AppendExecutionProvider_CUDA(0);
                        rtb1.AppendText("✓ ONNX using CUDA GPU\n");
                    }
                    catch
                    {
                        rtb1.AppendText("⚠ ONNX using CPU\n");
                    }
                }

                _onnxSession = new InferenceSession(modelPath, sessionOptions);
                rtb1.AppendText("✓ ONNX session initialized\n");
            }
            catch (Exception ex)
            {
                // Fallback to basic session
                _onnxSession = new InferenceSession(modelPath);
                rtb1.AppendText($"⚠ ONNX basic session: {ex.Message}\n");
            }
        }

        private VideoCapture CreateCameraCapture(int cameraIndex)
        {
            try
            {
                string url = _cameraUrls[cameraIndex];
                string cameraName = _cameraNames[cameraIndex];

                LogMessage($"Connecting to {cameraName}...");

                VideoCapture newCapture = new VideoCapture(url, VideoCapture.API.Ffmpeg);
                newCapture.Set(CapProp.Buffersize, 5);
                newCapture.Set(CapProp.Fps, 15);

                using (Mat test = new Mat())
                {
                    if (newCapture.Read(test) && !test.IsEmpty && test.Width > 0)
                    {
                        LogMessage($"✓ Connected to {cameraName}: {test.Width}x{test.Height}");
                        return newCapture;
                    }
                }

                LogMessage($"✗ Failed to connect to {cameraName}");
                newCapture?.Dispose();
                return null;
            }
            catch (Exception ex)
            {
                LogMessage($"Connection error for camera {cameraIndex}: {ex.Message}");
                return null;
            }
        }

        private void LogMessage(string message)
        {
            if (this.InvokeRequired)
            {
                this.BeginInvoke((MethodInvoker)(() => rtb1.AppendText(message + "\n")));
            }
            else
            {
                rtb1.AppendText(message + "\n");
            }
        }

        private void UpdateCameraIndicator(int cameraIndex)
        {
            if (_cameraIndicatorLabel == null || _cameraIndicatorLabel.IsDisposed) return;

            Action updateAction = () =>
            {
                _cameraIndicatorLabel.Text = $"📷 {_cameraNames[cameraIndex]}";
                _cameraIndicatorLabel.BackColor = cameraIndex == 0 ? Color.DarkGreen : Color.DarkBlue;
            };

            if (_cameraIndicatorLabel.InvokeRequired)
            {
                _cameraIndicatorLabel.BeginInvoke(updateAction);
            }
            else
            {
                updateAction();
            }
        }

        private void CameraSwitchTimer_Tick(object sender, EventArgs e)
        {
            if (_running && !_isSwitchingCamera)
            {
                int nextIndex = (_currentCameraIndex + 1) % _cameraUrls.Length;
                SwitchToCamera(nextIndex);
            }
        }

        private void SwitchToCamera(int targetCameraIndex)
        {
            if (_isSwitchingCamera) return;
            if (targetCameraIndex == _currentCameraIndex) return;

            Task.Run(() =>
            {
                _isSwitchingCamera = true;

                try
                {
                    LogMessage($"🔄 Switching to {_cameraNames[targetCameraIndex]}...");

                    VideoCapture newCapture = CreateCameraCapture(targetCameraIndex);

                    if (newCapture != null)
                    {
                        VideoCapture oldCapture;
                        lock (_captureLock)
                        {
                            oldCapture = _capture;
                            _capture = newCapture;
                            _currentCameraIndex = targetCameraIndex;
                            _resized = false;
                        }

                        if (oldCapture != null)
                        {
                            try
                            {
                                oldCapture.Stop();
                                oldCapture.Dispose();
                            }
                            catch { }
                        }

                        UpdateCameraIndicator(targetCameraIndex);
                        LogMessage($"✓ Now viewing {_cameraNames[targetCameraIndex]}");
                    }
                    else
                    {
                        LogMessage($"⚠ Failed to switch to {_cameraNames[targetCameraIndex]}, staying on current camera");
                    }
                }
                catch (Exception ex)
                {
                    LogMessage($"Switch error: {ex.Message}");
                }
                finally
                {
                    _isSwitchingCamera = false;
                }
            });
        }

        private void StartCapture(object streamUrl)
        {
            if (_net == null)
                LoadYolo();

            try
            {
                rtb1.AppendText("Starting multi-camera capture...\n");

                _capture = CreateCameraCapture(0);
                if (_capture == null)
                {
                    _capture = CreateCameraCapture(1);
                    if (_capture != null)
                    {
                        _currentCameraIndex = 1;
                    }
                }

                if (_capture == null)
                {
                    MessageBox.Show("Cannot connect to any camera.");
                    return;
                }

                UpdateCameraIndicator(_currentCameraIndex);

                _latestFrame = new Mat();
                _cancellationTokenSource = new CancellationTokenSource();
                _running = true;

                _cameraSwitchTimer = new System.Windows.Forms.Timer();
                _cameraSwitchTimer.Interval = CAMERA_SWITCH_INTERVAL_MS;
                _cameraSwitchTimer.Tick += CameraSwitchTimer_Tick;
                _cameraSwitchTimer.Start();

                rtb1.AppendText($"Camera switch timer started ({CAMERA_SWITCH_INTERVAL_MS / 1000}s intervals)\n");

                Task.Run(() => MultiCameraFrameLoop(_cancellationTokenSource.Token));

                if (_detectionEnabled)
                {
                    StartDetectionThread();
                }

                StartUploadThread();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Start error: {ex.Message}");
            }
        }

        private async Task MultiCameraFrameLoop(CancellationToken cancellationToken)
        {
            Bitmap previousBitmap = null;
            int frameCount = 0;
            int skipFrames = 2;
            int currentSkip = 0;
            int consecutiveFailures = 0;
            const int MAX_CONSECUTIVE_FAILURES = 30;

            while (!cancellationToken.IsCancellationRequested && _running)
            {
                try
                {
                    VideoCapture currentCapture;
                    int currentCamIndex;

                    lock (_captureLock)
                    {
                        currentCapture = _capture;
                        currentCamIndex = _currentCameraIndex;
                    }

                    if (currentCapture == null || _isSwitchingCamera)
                    {
                        await Task.Delay(50, cancellationToken);
                        continue;
                    }

                    using (Mat frame = new Mat())
                    {
                        bool grabbed = false;

                        try
                        {
                            grabbed = currentCapture.Grab();
                        }
                        catch
                        {
                            grabbed = false;
                        }

                        if (grabbed)
                        {
                            consecutiveFailures = 0;
                            currentSkip++;

                            if (currentSkip >= skipFrames)
                            {
                                currentSkip = 0;

                                bool retrieved = false;
                                try
                                {
                                    retrieved = currentCapture.Retrieve(frame);
                                }
                                catch
                                {
                                    retrieved = false;
                                }

                                if (retrieved && !frame.IsEmpty)
                                {
                                    frameCount++;

                                    if (!_resized)
                                    {
                                        LogMessage($"Stream [{_cameraNames[currentCamIndex]}]: {frame.Width}x{frame.Height}");
                                        _resized = true;
                                    }

                                    if (frameCount % 10 == 0)
                                    {
                                        lock (_frameLock)
                                        {
                                            _latestFrame?.Dispose();
                                            _latestFrame = frame.Clone();
                                        }
                                    }

                                    using (Mat displayFrame = frame.Clone())
                                    {
                                        DrawDetections(displayFrame);

                                        string cameraText = $"CAM {currentCamIndex + 1}: {_cameraNames[currentCamIndex]}";
                                        CvInvoke.PutText(displayFrame, cameraText,
                                            new Point(50, displayFrame.Height - 20), FontFace.HersheySimplex, 1.8,
                                            new MCvScalar(0, 255, 255), 3, LineType.AntiAlias);

                                        string timeText = DateTime.Now.ToString("HH:mm:ss");
                                        CvInvoke.PutText(displayFrame, timeText,
                                            new Point(10, 60), FontFace.HersheySimplex, 0.6,
                                            new MCvScalar(255, 255, 255), 2, LineType.AntiAlias);

                                        using (Mat resizedMat = new Mat())
                                        {
                                            CvInvoke.Resize(displayFrame, resizedMat, DISPLAY_RESOLUTION, interpolation: Inter.Area);

                                            Bitmap newBitmap = resizedMat.ToBitmap();

                                            this.BeginInvoke((MethodInvoker)(() =>
                                            {
                                                if (pictureBox1 != null && !pictureBox1.IsDisposed)
                                                {
                                                    var oldImage = pictureBox1.Image;
                                                    pictureBox1.Image = newBitmap;
                                                    oldImage?.Dispose();
                                                }
                                                else
                                                {
                                                    newBitmap?.Dispose();
                                                }
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            consecutiveFailures++;

                            if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES && !_isSwitchingCamera)
                            {
                                LogMessage($"Stream lost on {_cameraNames[currentCamIndex]}, attempting to switch...");
                                int nextIndex = (currentCamIndex + 1) % _cameraUrls.Length;
                                SwitchToCamera(nextIndex);
                                consecutiveFailures = 0;
                            }
                        }
                    }

                    await Task.Delay(15, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    LogMessage($"Frame error: {ex.Message}");
                    await Task.Delay(100, cancellationToken);
                }
            }

            previousBitmap?.Dispose();
        }

        private void DrawDetections(Mat frame)
        {
            List<string> _keep;
            string keepObjects = "keepObjects.txt";
            _keep = File.ReadAllLines(keepObjects).ToList();

            while (_detectionResults.TryDequeue(out DetectionResult result))
            {
                CvInvoke.Rectangle(frame, result.Rect, new MCvScalar(100, 255, 0), 4);

                string text = $"{result.Label} ({result.Confidence:F2})";
                int baseline = 0;
                var textSize = CvInvoke.GetTextSize(text, FontFace.HersheySimplex, 0.9, 2, ref baseline);

                Point textOrg = new Point(result.Rect.Left, result.Rect.Top - 5);
                Rectangle textBg = new Rectangle(textOrg.X, textOrg.Y - textSize.Height - 5,
                                                textSize.Width, textSize.Height + 10);

                CvInvoke.Rectangle(frame, textBg, new MCvScalar(0, 255, 0), -1);
                CvInvoke.PutText(frame, text, textOrg, FontFace.HersheySimplex, 1.6,
                               new MCvScalar(0, 0, 0), 3, LineType.AntiAlias);
                if (_keep.Contains(result.Label))
                {
                    Task.Run(() =>
                    {
                        try
                        {
                            SaveDetectedObject(result);
                        }
                        finally
                        {
                            result.FrameSnapshot?.Dispose();
                        }
                    });
                }
            }
        }

        private void StartDetectionThread()
        {
            _detectionTask = Task.Run(() => DetectionLoop(_cancellationTokenSource.Token));
        }

        private async Task DetectionLoop(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested && _running)
            {
                try
                {
                    if (!_detectionEnabled || _isSwitchingCamera)
                    {
                        await Task.Delay(100, cancellationToken);
                        continue;
                    }

                    Mat frameToProcess = null;
                    int cameraIndex;

                    lock (_frameLock)
                    {
                        cameraIndex = _currentCameraIndex;
                        if (_latestFrame != null && !_latestFrame.IsEmpty)
                        {
                            _frameSkipCounter++;
                            if (_frameSkipCounter >= FRAME_SKIP)
                            {
                                frameToProcess = _latestFrame.Clone();
                                _frameSkipCounter = 0;
                            }
                        }
                    }

                    if (frameToProcess != null)
                    {
                        await Task.Run(() => RunDetection(frameToProcess, cameraIndex), cancellationToken);
                        frameToProcess.Dispose();
                    }

                    await Task.Delay(500, cancellationToken);  // Reduced from 1000ms for more responsive detection
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    LogMessage($"Detection error: {ex.Message}");
                }
            }
        }

        // ============ MAIN DETECTION METHOD - NOW WITH TILED DETECTION ============
        private void RunDetection(Mat frame, int cameraIndex)
        {
            try
            {
                if (ENABLE_TILED_DETECTION && frame.Width > TILE_SIZE && frame.Height > TILE_SIZE)
                {
                    // Use tiled detection for high-resolution frames
                    RunTiledDetection(frame, cameraIndex);
                }
                else
                {
                    // Fall back to single-pass detection for smaller frames
                    RunSinglePassDetection(frame, cameraIndex);
                }
            }
            catch (Exception ex)
            {
                LogMessage($"Detection error: {ex.Message}");
            }
        }

        // ============ TILED DETECTION FOR SMALL OBJECTS ============
        private void RunTiledDetection(Mat frame, int cameraIndex)
        {
            List<DetectionResult> allDetections = new List<DetectionResult>();
            int stride = TILE_SIZE - TILE_OVERLAP;

            // Calculate number of tiles needed
            int tilesX = (int)Math.Ceiling((double)(frame.Width - TILE_OVERLAP) / stride);
            int tilesY = (int)Math.Ceiling((double)(frame.Height - TILE_OVERLAP) / stride);

            LogMessage($"🔍 Tiled detection: {tilesX}x{tilesY} tiles on {frame.Width}x{frame.Height} frame");

            // Process each tile
            for (int ty = 0; ty < tilesY; ty++)
            {
                for (int tx = 0; tx < tilesX; tx++)
                {
                    int x = Math.Min(tx * stride, frame.Width - TILE_SIZE);
                    int y = Math.Min(ty * stride, frame.Height - TILE_SIZE);

                    x = Math.Max(0, x);
                    y = Math.Max(0, y);

                    int width = Math.Min(TILE_SIZE, frame.Width - x);
                    int height = Math.Min(TILE_SIZE, frame.Height - y);

                    if (width < TILE_SIZE || height < TILE_SIZE)
                    {
                        // Handle edge tiles that need padding
                        using (Mat tile = new Mat(TILE_SIZE, TILE_SIZE, frame.Depth, frame.NumberOfChannels))
                        {
                            tile.SetTo(new MCvScalar(0, 0, 0));  // Black padding

                            Rectangle srcRoi = new Rectangle(x, y, width, height);
                            Rectangle dstRoi = new Rectangle(0, 0, width, height);

                            using (Mat srcRegion = new Mat(frame, srcRoi))
                            using (Mat dstRegion = new Mat(tile, dstRoi))
                            {
                                srcRegion.CopyTo(dstRegion);
                            }

                            RunDetectionOnTile(tile, x, y, TILE_SIZE, TILE_SIZE, frame, cameraIndex, allDetections);
                        }
                    }
                    else
                    {
                        // Normal tile extraction
                        Rectangle roi = new Rectangle(x, y, TILE_SIZE, TILE_SIZE);
                        using (Mat tile = new Mat(frame, roi))
                        {
                            RunDetectionOnTile(tile, x, y, TILE_SIZE, TILE_SIZE, frame, cameraIndex, allDetections);
                        }
                    }
                }
            }

            // Also run detection on full frame (downscaled) to catch large objects
            RunFullFrameDetection(frame, cameraIndex, allDetections);

            // Apply global NMS to remove duplicate detections across tiles
            ApplyGlobalNMS(allDetections, frame, cameraIndex);
        }

        private void RunDetectionOnTile(Mat tile, int offsetX, int offsetY, int tileWidth, int tileHeight,
            Mat fullFrame, int cameraIndex, List<DetectionResult> detections)
        {
            try
            {
                // Prepare input tensor
                Mat resized = new Mat();
                CvInvoke.Resize(tile, resized, new Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE));

                Mat rgb = new Mat();
                CvInvoke.CvtColor(resized, rgb, ColorConversion.Bgr2Rgb);

                float[] inputArray = new float[1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE];
                byte[] data = new byte[rgb.Width * rgb.Height * rgb.NumberOfChannels];
                System.Runtime.InteropServices.Marshal.Copy(rgb.DataPointer, data, 0, data.Length);

                int idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < YOLO_INPUT_SIZE; h++)
                    {
                        for (int w = 0; w < YOLO_INPUT_SIZE; w++)
                        {
                            int pixelIdx = (h * YOLO_INPUT_SIZE + w) * 3 + c;
                            inputArray[idx++] = data[pixelIdx] / 255.0f;
                        }
                    }
                }

                // Run inference using cached session
                lock (_sessionLock)
                {
                    if (_onnxSession != null)
                    {
                        var inputMeta = _onnxSession.InputMetadata;
                        var inputName = inputMeta.Keys.First();

                        var tensor = new DenseTensor<float>(inputArray, new[] { 1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE });
                        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

                        using (var results = _onnxSession.Run(inputs))
                        {
                            var output = results.First().AsEnumerable<float>().ToArray();
                            ProcessTileOutput(output, tileWidth, tileHeight, offsetX, offsetY,
                                fullFrame.Width, fullFrame.Height, cameraIndex, detections);
                        }
                    }
                }

                resized.Dispose();
                rgb.Dispose();
            }
            catch (Exception ex)
            {
                LogMessage($"Tile detection error: {ex.Message}");
            }
        }

        private void ProcessTileOutput(float[] output, int tileWidth, int tileHeight,
            int offsetX, int offsetY, int frameWidth, int frameHeight, int cameraIndex, List<DetectionResult> detections)
        {
            int numClasses = 600;
            int numDetections = 8400;

            float scaleX = (float)tileWidth / YOLO_INPUT_SIZE;
            float scaleY = (float)tileHeight / YOLO_INPUT_SIZE;

            // Use lower threshold for tiles to catch small objects
            float tileConfThreshold = CONFIDENCE_THRESHOLD * SMALL_OBJECT_CONFIDENCE_MULTIPLIER;

            for (int i = 0; i < numDetections; i++)
            {
                float centerX = output[i] * scaleX + offsetX;
                float centerY = output[numDetections + i] * scaleY + offsetY;
                float width = output[2 * numDetections + i] * scaleX;
                float height = output[3 * numDetections + i] * scaleY;

                float maxScore = 0;
                int maxClassId = 0;
                for (int j = 0; j < numClasses; j++)
                {
                    float score = output[(4 + j) * numDetections + i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = j;
                    }
                }

                // Apply size-aware confidence threshold
                float effectiveThreshold = GetSizeAwareThreshold(width, height, tileWidth, tileHeight, tileConfThreshold);

                if (maxScore >= effectiveThreshold)
                {
                    int left = (int)(centerX - width / 2);
                    int top = (int)(centerY - height / 2);

                    // Clamp to frame boundaries
                    left = Math.Max(0, Math.Min(left, frameWidth - 1));
                    top = Math.Max(0, Math.Min(top, frameHeight - 1));
                    int clampedWidth = Math.Min((int)width, frameWidth - left);
                    int clampedHeight = Math.Min((int)height, frameHeight - top);

                    if (clampedWidth > 10 && clampedHeight > 10)  // Filter out tiny noise
                    {
                        lock (detections)
                        {
                            detections.Add(new DetectionResult
                            {
                                Rect = new Rectangle(left, top, clampedWidth, clampedHeight),
                                Label = _labels[maxClassId],
                                Confidence = maxScore,
                                FrameSnapshot = null,  // Will be set after NMS
                                CameraIndex = cameraIndex
                            });
                        }
                    }
                }
            }
        }

        // Get confidence threshold based on object size (lower for small objects)
        private float GetSizeAwareThreshold(float width, float height, int tileWidth, int tileHeight, float baseThreshold)
        {
            float area = width * height;
            float tileArea = tileWidth * tileHeight;
            float relativeSize = area / tileArea;

            if (relativeSize < 0.005f)       // Very small objects (< 0.5% of tile)
                return baseThreshold * 0.6f;
            else if (relativeSize < 0.02f)   // Small objects (< 2% of tile)
                return baseThreshold * 0.75f;
            else if (relativeSize < 0.05f)   // Medium-small objects (< 5% of tile)
                return baseThreshold * 0.9f;

            return baseThreshold;
        }

        // Run detection on full downscaled frame to catch large objects
        private void RunFullFrameDetection(Mat frame, int cameraIndex, List<DetectionResult> detections)
        {
            try
            {
                Mat resized = new Mat();
                CvInvoke.Resize(frame, resized, new Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE));

                Mat rgb = new Mat();
                CvInvoke.CvtColor(resized, rgb, ColorConversion.Bgr2Rgb);

                float[] inputArray = new float[1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE];
                byte[] data = new byte[rgb.Width * rgb.Height * rgb.NumberOfChannels];
                System.Runtime.InteropServices.Marshal.Copy(rgb.DataPointer, data, 0, data.Length);

                int idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < YOLO_INPUT_SIZE; h++)
                    {
                        for (int w = 0; w < YOLO_INPUT_SIZE; w++)
                        {
                            int pixelIdx = (h * YOLO_INPUT_SIZE + w) * 3 + c;
                            inputArray[idx++] = data[pixelIdx] / 255.0f;
                        }
                    }
                }

                lock (_sessionLock)
                {
                    if (_onnxSession != null)
                    {
                        var inputMeta = _onnxSession.InputMetadata;
                        var inputName = inputMeta.Keys.First();

                        var tensor = new DenseTensor<float>(inputArray, new[] { 1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE });
                        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

                        using (var results = _onnxSession.Run(inputs))
                        {
                            var output = results.First().AsEnumerable<float>().ToArray();
                            ProcessFullFrameOutput(output, frame.Width, frame.Height, cameraIndex, detections);
                        }
                    }
                }

                resized.Dispose();
                rgb.Dispose();
            }
            catch (Exception ex)
            {
                LogMessage($"Full frame detection error: {ex.Message}");
            }
        }

        private void ProcessFullFrameOutput(float[] output, int originalWidth, int originalHeight,
            int cameraIndex, List<DetectionResult> detections)
        {
            int numClasses = 600;
            int numDetections = 8400;

            float scaleX = (float)originalWidth / YOLO_INPUT_SIZE;
            float scaleY = (float)originalHeight / YOLO_INPUT_SIZE;

            for (int i = 0; i < numDetections; i++)
            {
                float centerX = output[i] * scaleX;
                float centerY = output[numDetections + i] * scaleY;
                float width = output[2 * numDetections + i] * scaleX;
                float height = output[3 * numDetections + i] * scaleY;

                float maxScore = 0;
                int maxClassId = 0;
                for (int j = 0; j < numClasses; j++)
                {
                    float score = output[(4 + j) * numDetections + i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = j;
                    }
                }

                // Use standard threshold for full-frame detection (catches larger objects)
                if (maxScore >= CONFIDENCE_THRESHOLD)
                {
                    int left = (int)(centerX - width / 2);
                    int top = (int)(centerY - height / 2);

                    left = Math.Max(0, Math.Min(left, originalWidth - 1));
                    top = Math.Max(0, Math.Min(top, originalHeight - 1));

                    lock (detections)
                    {
                        detections.Add(new DetectionResult
                        {
                            Rect = new Rectangle(left, top, (int)width, (int)height),
                            Label = _labels[maxClassId],
                            Confidence = maxScore,
                            FrameSnapshot = null,
                            CameraIndex = cameraIndex
                        });
                    }
                }
            }
        }

        // Apply global NMS across all detections from all tiles
        private void ApplyGlobalNMS(List<DetectionResult> detections, Mat sourceFrame, int cameraIndex)
        {
            if (detections.Count == 0) return;

            // Group detections by class
            var groupedByClass = detections.GroupBy(d => d.Label);

            foreach (var group in groupedByClass)
            {
                var boxes = group.Select(d => d.Rect).ToArray();
                var confidences = group.Select(d => d.Confidence).ToArray();
                var items = group.ToList();

                if (boxes.Length == 0) continue;

                var indices = DnnInvoke.NMSBoxes(boxes, confidences,
                    CONFIDENCE_THRESHOLD * SMALL_OBJECT_CONFIDENCE_MULTIPLIER, NMS_THRESHOLD);

                if (indices != null && indices.Length > 0)
                {
                    foreach (int idx in indices)
                    {
                        if (idx >= 0 && idx < items.Count)
                        {
                            var detection = items[idx];
                            detection.FrameSnapshot = sourceFrame.Clone();
                            _detectionResults.Enqueue(detection);
                            LogMessage($"✓ [{_cameraNames[cameraIndex]}] {detection.Label} ({detection.Confidence:F2}) at ({detection.Rect.X},{detection.Rect.Y})");
                        }
                    }
                }
            }
        }

        // Fallback single-pass detection for smaller frames
        private void RunSinglePassDetection(Mat frame, int cameraIndex)
        {
            try
            {
                Mat resized = new Mat();
                CvInvoke.Resize(frame, resized, new Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE));

                Mat rgb = new Mat();
                CvInvoke.CvtColor(resized, rgb, ColorConversion.Bgr2Rgb);

                float[] inputArray = new float[1 * 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE];
                byte[] data = new byte[rgb.Width * rgb.Height * rgb.NumberOfChannels];
                System.Runtime.InteropServices.Marshal.Copy(rgb.DataPointer, data, 0, data.Length);

                int idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < YOLO_INPUT_SIZE; h++)
                    {
                        for (int w = 0; w < YOLO_INPUT_SIZE; w++)
                        {
                            int pixelIdx = (h * YOLO_INPUT_SIZE + w) * 3 + c;
                            inputArray[idx++] = data[pixelIdx] / 255.0f;
                        }
                    }
                }

                lock (_sessionLock)
                {
                    if (_onnxSession != null)
                    {
                        var inputMeta = _onnxSession.InputMetadata;
                        var inputName = inputMeta.Keys.First();

                        var tensor = new DenseTensor<float>(
                            inputArray,
                            new[] { 1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE }
                        );

                        var inputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor(inputName, tensor)
                        };

                        using (var results = _onnxSession.Run(inputs))
                        {
                            var output = results.First().AsEnumerable<float>().ToArray();
                            ProcessYOLOv11OutputOnnx(output, frame.Width, frame.Height, frame, cameraIndex);
                        }
                    }
                }

                resized.Dispose();
                rgb.Dispose();
            }
            catch (Exception ex)
            {
                LogMessage($"Detection error: {ex.Message}");
            }
        }

        private void ProcessYOLOv11OutputOnnx(float[] output, int originalWidth, int originalHeight, Mat sourceFrame, int cameraIndex)
        {
            try
            {
                int numClasses = 600;
                int numDetections = 8400;

                float scaleX = (float)originalWidth / YOLO_INPUT_SIZE;
                float scaleY = (float)originalHeight / YOLO_INPUT_SIZE;

                List<Rectangle> boxes = new List<Rectangle>();
                List<float> confidences = new List<float>();
                List<int> classIds = new List<int>();

                for (int i = 0; i < numDetections; i++)
                {
                    float centerX = output[i] * scaleX;
                    float centerY = output[numDetections + i] * scaleY;
                    float width = output[2 * numDetections + i] * scaleX;
                    float height = output[3 * numDetections + i] * scaleY;

                    float maxScore = 0;
                    int maxClassId = 0;
                    for (int j = 0; j < numClasses; j++)
                    {
                        float score = output[(4 + j) * numDetections + i];
                        if (score > maxScore)
                        {
                            maxScore = score;
                            maxClassId = j;
                        }
                    }

                    if (maxScore >= CONFIDENCE_THRESHOLD)
                    {
                        int left = (int)(centerX - width / 2);
                        int top = (int)(centerY - height / 2);

                        boxes.Add(new Rectangle(left, top, (int)width, (int)height));
                        confidences.Add(maxScore);
                        classIds.Add(maxClassId);
                    }
                }

                if (boxes.Count > 0)
                {
                    var indices = DnnInvoke.NMSBoxes(
                        boxes.ToArray(),
                        confidences.ToArray(),
                        CONFIDENCE_THRESHOLD,
                        NMS_THRESHOLD
                    );

                    if (indices != null && indices.Length > 0)
                    {
                        foreach (int idx in indices)
                        {
                            var rect = boxes[idx];
                            string label = _labels[classIds[idx]];

                            var detection = new DetectionResult
                            {
                                Rect = rect,
                                Label = label,
                                Confidence = confidences[idx],
                                FrameSnapshot = sourceFrame.Clone(),
                                CameraIndex = cameraIndex
                            };

                            _detectionResults.Enqueue(detection);

                            LogMessage($"✓ [{_cameraNames[cameraIndex]}] {label} ({confidences[idx]:F2})");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogMessage($"ERROR: {ex.Message}");
            }
        }

        private void SaveDetectedObject(DetectionResult detection)
        {
            try
            {
                if (detection?.FrameSnapshot == null || detection.FrameSnapshot.IsEmpty)
                {
                    detection?.FrameSnapshot?.Dispose();
                    return;
                }

                if (IsDuplicateDetection(detection))
                {
                    return;
                }

                TrackDetection(detection);

                using (Mat fullFrame = detection.FrameSnapshot.Clone())
                {
                    CvInvoke.Rectangle(fullFrame, detection.Rect, new MCvScalar(0, 255, 0), 4);

                    string label = $"{detection.Label} ({detection.Confidence:F2})";
                    CvInvoke.PutText(fullFrame, label,
                        new Point(detection.Rect.Left, detection.Rect.Top - 20),
                        FontFace.HersheySimplex, 2, new MCvScalar(0, 255, 0), 3);

                    string cameraPrefix = $"";
                    string filename = $"{cameraPrefix}{detection.Label}_{DateTime.Now:MM-dd-yyyy_HHmmss}.jpg";
                    string fullPath = Path.Combine(_photoDirectory, filename);

                    var compressionParams = new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.JpegQuality, JPEG_QUALITY);
                    CvInvoke.Imwrite(fullPath, fullFrame, compressionParams);

                    lock (_uploadQueue)
                    {
                        _uploadQueue.Enqueue(fullPath);
                    }

                    var fileInfo = new FileInfo(fullPath);
                    LogMessage($"✓ Saved: {filename} ({fileInfo.Length / 1024}KB)");
                }

                SaveCroppedVersion(detection);
            }
            catch (Exception ex)
            {
                LogMessage($"Save error: {ex.Message}");
            }
        }

        private void SaveCroppedVersion(DetectionResult detection)
        {
            int paddingX = Math.Max(200, (int)(detection.Rect.Width * 0.75));
            int paddingY = Math.Max(200, (int)(detection.Rect.Height * 0.75));

            Rectangle roi = new Rectangle(
                Math.Max(0, detection.Rect.X - paddingX),
                Math.Max(0, detection.Rect.Y - paddingY),
                Math.Min(detection.FrameSnapshot.Width - Math.Max(0, detection.Rect.X - paddingX),
                         detection.Rect.Width + (paddingX * 2)),
                Math.Min(detection.FrameSnapshot.Height - Math.Max(0, detection.Rect.Y - paddingY),
                         detection.Rect.Height + (paddingY * 2))
            );

            if (roi.Width > 0 && roi.Height > 0)
            {
                using (Mat croppedImage = new Mat(detection.FrameSnapshot, roi))
                {
                    string cameraPrefix = $"CAM{detection.CameraIndex + 1}_";
                    string filename = $"{detection.Label}_crop_{DateTime.Now:MM-dd-yyyy_HHmmss}.jpg";
                    string fullPath = Path.Combine(_photoDirectory, filename);

                    var compressionParams = new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.JpegQuality, JPEG_QUALITY);
                    CvInvoke.Imwrite(fullPath, croppedImage, compressionParams);

                    lock (_uploadQueue)
                    {
                        _uploadQueue.Enqueue(fullPath);
                    }
                }
            }
        }

        private bool IsDuplicateDetection(DetectionResult detection)
        {
            string objectType = $"CAM{detection.CameraIndex}_{detection.Label}";

            CleanupOldTrackedObjects();

            lock (_trackedObjects)
            {
                if (!_trackedObjects.ContainsKey(objectType))
                {
                    _trackedObjects[objectType] = new List<TrackedObject>();
                    return false;
                }

                var trackedList = _trackedObjects[objectType];
                if (trackedList == null) return false;

                foreach (var tracked in trackedList)
                {
                    if (tracked != null && tracked.IsNearLocation(detection.Rect, LOCATION_TOLERANCE))
                    {
                        tracked.LastSeen = DateTime.Now;

                        var timeSinceFirst = DateTime.Now - tracked.FirstSeen;
                        if (timeSinceFirst.TotalSeconds < SAVE_INTERVAL_SECONDS)
                        {
                            return true;
                        }
                        else
                        {
                            tracked.FirstSeen = DateTime.Now;
                            return false;
                        }
                    }
                }
            }

            return false;
        }

        private void TrackDetection(DetectionResult detection)
        {
            string objectType = $"{detection.Label}";

            lock (_trackedObjects)
            {
                if (!_trackedObjects.ContainsKey(objectType))
                {
                    _trackedObjects[objectType] = new List<TrackedObject>();
                }

                _trackedObjects[objectType].Add(new TrackedObject
                {
                    Location = detection.Rect,
                    FirstSeen = DateTime.Now,
                    LastSeen = DateTime.Now,
                    UniqueId = Guid.NewGuid().ToString()
                });
            }
        }

        private void CleanupOldTrackedObjects()
        {
            try
            {
                var cutoffTime = DateTime.Now.AddMinutes(-TRACKING_TIMEOUT_MINUTES);

                lock (_trackedObjects)
                {
                    var keysToRemove = new List<string>();

                    foreach (var kvp in _trackedObjects)
                    {
                        if (kvp.Value == null)
                        {
                            keysToRemove.Add(kvp.Key);
                            continue;
                        }

                        kvp.Value.RemoveAll(t => t == null || t.LastSeen < cutoffTime);

                        if (kvp.Value.Count == 0)
                        {
                            keysToRemove.Add(kvp.Key);
                        }
                    }

                    foreach (var key in keysToRemove)
                    {
                        _trackedObjects.Remove(key);
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"CleanupOldTrackedObjects error: {ex.Message}");
            }
        }

        private void StartUploadThread()
        {
            if (_uploadTask == null || _uploadTask.IsCompleted)
            {
                _uploadTask = Task.Run(() => ProcessUploadQueue());
            }
        }

        private async Task ProcessUploadQueue()
        {
            while (_running)
            {
                try
                {
                    string fileToUpload = null;

                    lock (_uploadQueue)
                    {
                        if (_uploadQueue.Count > 0)
                        {
                            LogMessage($"📤 Queue has {_uploadQueue.Count} files waiting");
                        }
                        if (_uploadQueue.Count > 0)
                        {
                            fileToUpload = _uploadQueue.Dequeue();
                        }
                    }

                    if (fileToUpload != null && File.Exists(fileToUpload))
                    {
                        string fileName = Path.GetFileName(fileToUpload);
                        if (!_uploadedFiles.Contains(fileName))
                        {
                            bool uploaded = await UploadFileToFTP(fileToUpload);
                            if (uploaded)
                            {
                                _uploadedFiles.Add(fileName);
                            }
                        }
                    }

                    await Task.Delay(1000);
                }
                catch (Exception ex)
                {
                    LogMessage($"Upload error: {ex.Message}");
                }
            }
        }

        private async Task<bool> UploadFileToFTP(string localFilePath)
        {
            if (string.IsNullOrEmpty(_ftpHost) || string.IsNullOrEmpty(_ftpUsername))
            {
                LogMessage("FTP not configured (FTP_HOST / FTP_USERNAME / FTP_PASSWORD); skipping upload.");
                return false;
            }

            try
            {
                string fileName = Path.GetFileName(localFilePath);
                string ftpUrl = $"ftp://{_ftpHost}{_ftpPath}{fileName}";

                FtpWebRequest request = (FtpWebRequest)WebRequest.Create(ftpUrl);
                request.Method = WebRequestMethods.Ftp.UploadFile;
                request.Credentials = new NetworkCredential(_ftpUsername, _ftpPassword);
                request.UsePassive = true;
                request.UseBinary = true;
                request.KeepAlive = false;

                using (FileStream fileStream = File.OpenRead(localFilePath))
                using (System.IO.Stream requestStream = await request.GetRequestStreamAsync())
                {
                    await fileStream.CopyToAsync(requestStream);
                }

                using (FtpWebResponse response = (FtpWebResponse)await request.GetResponseAsync())
                {
                    if (response.StatusCode == FtpStatusCode.ClosingData ||
                        response.StatusCode == FtpStatusCode.FileActionOK)
                    {
                        LogMessage($"✓ Uploaded: {fileName}");
                        return true;
                    }
                }

                return false;
            }
            catch (Exception ex)
            {
                LogMessage($"FTP error: {ex.Message}");
                return false;
            }
        }

        private void StopCapture()
        {
            try
            {
                _running = false;
                _detectionEnabled = false;
                _cancellationTokenSource?.Cancel();

                if (_cameraSwitchTimer != null)
                {
                    _cameraSwitchTimer.Stop();
                    _cameraSwitchTimer.Tick -= CameraSwitchTimer_Tick;
                    _cameraSwitchTimer.Dispose();
                    _cameraSwitchTimer = null;
                }

                Task.Delay(200).Wait();

                lock (_captureLock)
                {
                    _capture?.Stop();
                    _capture?.Dispose();
                    _capture = null;
                }

                // Dispose ONNX session
                lock (_sessionLock)
                {
                    _onnxSession?.Dispose();
                    _onnxSession = null;
                }

                try { _detectionTask?.Wait(1000); } catch { }
                try { _uploadTask?.Wait(5000); } catch { }

                lock (_frameLock)
                {
                    _latestFrame?.Dispose();
                    _latestFrame = null;
                }

                if (pictureBox1 != null && !pictureBox1.IsDisposed)
                {
                    if (pictureBox1.InvokeRequired)
                    {
                        pictureBox1.Invoke((MethodInvoker)(() =>
                        {
                            pictureBox1.Image?.Dispose();
                            pictureBox1.Image = null;
                        }));
                    }
                    else
                    {
                        pictureBox1.Image?.Dispose();
                        pictureBox1.Image = null;
                    }
                }

                if (_cameraIndicatorLabel != null && !_cameraIndicatorLabel.IsDisposed)
                {
                    Action resetAction = () =>
                    {
                        _cameraIndicatorLabel.Text = "📷 No camera connected";
                        _cameraIndicatorLabel.BackColor = Color.Gray;
                    };

                    if (_cameraIndicatorLabel.InvokeRequired)
                        _cameraIndicatorLabel.Invoke(resetAction);
                    else
                        resetAction();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error stopping capture: {ex.Message}");
            }
            finally
            {
                _resized = false;
                _currentCameraIndex = 0;
                _trackedObjects.Clear();
                _lastSaveTime.Clear();
                _uploadQueue.Clear();
                while (_detectionResults.TryDequeue(out _)) { }
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            LoadYolo();

            this.WindowState = FormWindowState.Maximized;

            pictureBox1 = new PictureBox();
            pictureBox1.Location = new Point(12, 12);
            pictureBox1.Size = new Size(1280, 720);
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox1.BackColor = Color.Black;
            pictureBox1.BorderStyle = BorderStyle.FixedSingle;
            this.Controls.Add(pictureBox1);

            _cameraIndicatorLabel = new Label
            {
                Location = new Point(12, 740),
                Size = new Size(300, 50),
                Text = "📷 No camera connected",
                Font = new Font("Segoe UI", 11, FontStyle.Bold),
                ForeColor = Color.White,
                BackColor = Color.Gray,
                TextAlign = ContentAlignment.MiddleCenter
            };
            this.Controls.Add(_cameraIndicatorLabel);
            _cameraIndicatorLabel.BringToFront();

            if (rtb1 != null)
            {
                rtb1.Location = new Point(1310, 12);
                rtb1.Size = new Size(400, 700);
                rtb1.BringToFront();
            }

            if (button1 != null)
            {
                button1.Location = new Point(320, 740);
                button1.Size = new Size(100, 40);
                button1.Text = "START";
                button1.BringToFront();
            }

            if (button2 != null)
            {
                button2.Location = new Point(430, 740);
                button2.Size = new Size(100, 40);
                button2.Text = "STOP";
                button2.BringToFront();
            }

            Button btnSwitchCamera = new Button
            {
                Location = new Point(540, 740),
                Size = new Size(120, 40),
                Text = "SWITCH CAM",
                Font = new Font("Segoe UI", 9, FontStyle.Bold)
            };
            btnSwitchCamera.Click += (s, ev) =>
            {
                if (_running && !_isSwitchingCamera)
                {
                    int nextIndex = (_currentCameraIndex + 1) % _cameraUrls.Length;
                    SwitchToCamera(nextIndex);
                }
            };
            this.Controls.Add(btnSwitchCamera);
            btnSwitchCamera.BringToFront();

            trackBar1 = new TrackBar();
            trackBar1.Location = new Point(12, 800);
            trackBar1.Minimum = 10;
            trackBar1.Maximum = 90;
            trackBar1.Value = (int)(CONFIDENCE_THRESHOLD * 100);
            trackBar1.Size = new Size(600, 40);
            trackBar1.BringToFront();
            trackBar1.Scroll += (s, ev) =>
            {
                float newThreshold = trackBar1.Value / 100.0f;
                CONFIDENCE_THRESHOLD = newThreshold;
                tb1.Text = "Confidence Threshold set to  " + CONFIDENCE_THRESHOLD.ToString("F2");
            };
            this.Controls.Add(trackBar1);

            tb1 = new TextBox
            {
                Location = new Point(620, 800),
                Size = new Size(400, 30),
                Text = "Confidence Threshold set to  " + CONFIDENCE_THRESHOLD.ToString("F2")
            };
            this.Controls.Add(tb1);

            CheckBox chkDetection = new CheckBox
            {
                Text = "Enable Detection",
                Location = new Point(670, 750),
                AutoSize = true,
                Checked = true
            };
            chkDetection.CheckedChanged += (s, ev) =>
            {
                _detectionEnabled = chkDetection.Checked;
                if (_detectionEnabled && _running && (_detectionTask == null || _detectionTask.IsCompleted))
                {
                    StartDetectionThread();
                }
                rtb1.AppendText($"Detection {(_detectionEnabled ? "enabled" : "disabled")}\n");
            };
            this.Controls.Add(chkDetection);
            chkDetection.BringToFront();

            // Add checkbox for tiled detection toggle
            CheckBox chkTiled = new CheckBox
            {
                Text = "Tiled Detection (Small Objects)",
                Location = new Point(670, 775),
                AutoSize = true,
                Checked = ENABLE_TILED_DETECTION
            };
            chkTiled.CheckedChanged += (s, ev) =>
            {
                // Note: ENABLE_TILED_DETECTION is readonly, so we'd need to make it non-readonly
                // to support runtime toggling. For now, this is informational.
                rtb1.AppendText($"Tiled detection: {(chkTiled.Checked ? "ON" : "OFF")} (requires restart)\n");
            };
            this.Controls.Add(chkTiled);
            chkTiled.BringToFront();

            button2.Click += button2_Click;

            if (string.IsNullOrEmpty(_ftpUsername) || string.IsNullOrEmpty(_ftpPassword))
            {
                rtb1.AppendText("Warning: FTP credentials not set\n");
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (!_running)
            {
                StartCapture(0);
                rtb1.AppendText("Multi-camera stream started (10s intervals)\n");
                rtb1.AppendText($"Tiled detection: {(ENABLE_TILED_DETECTION ? "ENABLED" : "DISABLED")}\n");
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (_running)
            {
                StopCapture();
                rtb1.AppendText("Stream stopped\n");
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            rtb1.AppendText("=== System Info ===\n");

            try
            {
                bool hasCuda = CudaInvoke.HasCuda;
                rtb1.AppendText($"CUDA: {(hasCuda ? "Available" : "Not available")}\n");
                if (hasCuda)
                {
                    rtb1.AppendText($"CUDA Devices: {CudaInvoke.GetCudaEnabledDeviceCount()}\n");
                }
            }
            catch
            {
                rtb1.AppendText("CUDA: Not available\n");
            }

            try
            {
                bool hasOpenCL = CvInvoke.UseOpenCL;
                rtb1.AppendText($"OpenCL: {(hasOpenCL ? "Available" : "Not available")}\n");
            }
            catch
            {
                rtb1.AppendText("OpenCL: Not available\n");
            }

            rtb1.AppendText($"Model: YOLOv11x\n");
            rtb1.AppendText($"Input: {YOLO_INPUT_WIDTH}x{YOLO_INPUT_HEIGHT}\n");
            rtb1.AppendText($"Confidence: {CONFIDENCE_THRESHOLD}\n");
            rtb1.AppendText($"Tiled Detection: {(ENABLE_TILED_DETECTION ? "ON" : "OFF")}\n");
            rtb1.AppendText($"Tile Size: {TILE_SIZE}px, Overlap: {TILE_OVERLAP}px\n");
            rtb1.AppendText($"Small Object Threshold Multiplier: {SMALL_OBJECT_CONFIDENCE_MULTIPLIER}\n");
            rtb1.AppendText($"Cameras: {_cameraUrls.Length}\n");
            rtb1.AppendText($"Current Camera: {_cameraNames[_currentCameraIndex]}\n");
            rtb1.AppendText($"Switch Interval: {CAMERA_SWITCH_INTERVAL_MS / 1000}s\n");
            rtb1.AppendText("==================\n");
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            StopCapture();
            base.OnFormClosing(e);
        }
    }
}