import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

late List<CameraDescription> cameras;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: LiveClassifier());
  }
}

class LiveClassifier extends StatefulWidget {
  @override
  State<LiveClassifier> createState() => _LiveClassifierState();
}

class _LiveClassifierState extends State<LiveClassifier> {
  late CameraController _cameraController;
  late Interpreter _interpreter;
  List<String> _labels = [];
  bool _isProcessing = false;
  String _result = "Initializing...";

  @override
  void initState() {
    super.initState();
    loadModelAndLabels();
    initCamera();
  }

  Future<void> loadModelAndLabels() async {
    _interpreter = await Interpreter.fromAsset('model.tflite');
    _labels = (await rootBundle.loadString('assets/labels.txt'))
        .split('\n')
        .where((line) => line.trim().isNotEmpty)
        .toList();
  }

  void initCamera() async {
    _cameraController =
        CameraController(cameras[0], ResolutionPreset.low, enableAudio: false);

    await _cameraController.initialize();
    _cameraController.startImageStream((image) async {
      if (_isProcessing) return;
      _isProcessing = true;
      await classifyFrame(image);
      _isProcessing = false;
    });

    if (mounted) setState(() {});
  }

  Future<void> classifyFrame(CameraImage image) async {
    try {
      img.Image convertedImage = _convertCameraImage(image);
      img.Image resizedImage = img.copyResize(convertedImage, width: 224, height: 224);

      var input = imageToByteListFloat32(resizedImage, 224);
      var output = List.filled(_labels.length, 0.0).reshape([1, _labels.length]);

      _interpreter.run(input, output);

      final confidence = output[0];
      int topIndex = confidence.indexWhere((e) => e == confidence.reduce((a, b) => a > b ? a : b));
      String label = _labels[topIndex];

      setState(() {
        _result = "$label: ${(confidence[topIndex] * 100).toStringAsFixed(1)}%";
      });
    } catch (e) {
      print("Error in classification: $e");
    }
  }

  img.Image _convertCameraImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;
    final img.Image imgBuffer = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex =
            uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final int index = y * width + x;

        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        int r = (yp + (1.370705 * (vp - 128))).toInt();
        int g = (yp - (0.698001 * (vp - 128)) - (0.337633 * (up - 128))).toInt();
        int b = (yp + (1.732446 * (up - 128))).toInt();

        imgBuffer.setPixelRgba(x, y, r.clamp(0, 255), g.clamp(0, 255), b.clamp(0, 255), 255);
      }
    }
    return imgBuffer;
  }

  Float32List imageToByteListFloat32(img.Image image, int inputSize) {
    var buffer = Float32List(inputSize * inputSize * 3);
    int index = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = image.getPixel(x, y);
        buffer[index++] = pixel.r / 255.0;
        buffer[index++] = pixel.g / 255.0;
        buffer[index++] = pixel.b / 255.0;
      }
    }
    return buffer;
  }



  @override
  void dispose() {
    _cameraController.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Live Classification")),
      body: Column(
        children: [
          AspectRatio(
              aspectRatio: _cameraController.value.aspectRatio,
              child: CameraPreview(_cameraController)),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              _result,
              style: const TextStyle(fontSize: 20),
            ),
          ),
        ],
      ),
    );
  }
}
