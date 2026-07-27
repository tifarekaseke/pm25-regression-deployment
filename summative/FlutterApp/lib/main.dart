import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const AirQualityApp());
}

class AirQualityApp extends StatelessWidget {
  const AirQualityApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'PM2.5 Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          filled: true,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const String apiUrl =
      'http://192.168.1.83:8000/predict';

  final _formKey = GlobalKey<FormState>();
  final Map<String, TextEditingController> _controllers = {
    'month': TextEditingController(text: '7'),
    'hour': TextEditingController(text: '14'),
    'pm10': TextEditingController(text: '120'),
    'so2': TextEditingController(text: '12'),
    'no2': TextEditingController(text: '45'),
    'co': TextEditingController(text: '900'),
    'o3': TextEditingController(text: '70'),
    'temperature': TextEditingController(text: '28.5'),
    'pressure': TextEditingController(text: '1004'),
    'dew_point': TextEditingController(text: '17.2'),
    'rainfall': TextEditingController(text: '0'),
    'wind_speed': TextEditingController(text: '2.4'),
  };

  final List<String> _directions = const [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
  ];
  final List<String> _stations = const [
    'Aotizhongxin',
    'Changping',
    'Dingling',
    'Dongsi',
    'Guanyuan',
    'Gucheng',
    'Huairou',
    'Nongzhanguan',
    'Shunyi',
    'Tiantan',
    'Wanliu',
    'Wanshouxigong',
  ];

  String _windDirection = 'SE';
  String _station = 'Aotizhongxin';
  bool _isLoading = false;
  String? _result;
  bool _resultIsError = false;

  @override
  void dispose() {
    for (final controller in _controllers.values) {
      controller.dispose();
    }
    super.dispose();
  }

  String? _validateNumber(
    String? value,
    String label,
    double minimum,
    double maximum,
  ) {
    if (value == null || value.trim().isEmpty) {
      return '$label is required';
    }
    final parsed = double.tryParse(value.trim());
    if (parsed == null) return '$label must be a number';
    if (parsed < minimum || parsed > maximum) {
      return '$label must be between $minimum and $maximum';
    }
    return null;
  }

  Widget _numberField(
    String key,
    String label,
    double minimum,
    double maximum, {
    bool integer = false,
    String? suffix,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextFormField(
        controller: _controllers[key],
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          suffixText: suffix,
        ),
        validator: (value) {
          final rangeError = _validateNumber(value, label, minimum, maximum);
          if (rangeError != null) return rangeError;
          if (integer && int.tryParse(value!.trim()) == null) {
            return '$label must be a whole number';
          }
          return null;
        },
      ),
    );
  }

  String _extractApiError(http.Response response) {
    try {
      final decoded = jsonDecode(response.body);
      final detail = decoded['detail'];
      if (detail is String) return detail;
      if (detail is List) {
        return detail.map((item) => item['msg'] ?? item.toString()).join('\n');
      }
    } catch (_) {
      // Fall back to the response body below.
    }
    return response.body.isNotEmpty
        ? response.body
        : 'The server returned status ${response.statusCode}.';
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _result = null;
      _resultIsError = false;
    });

    final payload = {
      'month': int.parse(_controllers['month']!.text),
      'hour': int.parse(_controllers['hour']!.text),
      'pm10': double.parse(_controllers['pm10']!.text),
      'so2': double.parse(_controllers['so2']!.text),
      'no2': double.parse(_controllers['no2']!.text),
      'co': double.parse(_controllers['co']!.text),
      'o3': double.parse(_controllers['o3']!.text),
      'temperature': double.parse(_controllers['temperature']!.text),
      'pressure': double.parse(_controllers['pressure']!.text),
      'dew_point': double.parse(_controllers['dew_point']!.text),
      'rainfall': double.parse(_controllers['rainfall']!.text),
      'wind_speed': double.parse(_controllers['wind_speed']!.text),
      'wind_direction': _windDirection,
      'station': _station,
    };

    try {
      final response = await http
          .post(
            Uri.parse(apiUrl),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 45));

      if (!mounted) return;
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _result =
              '${decoded['predicted_pm25']} ${decoded['unit']}\nModel: ${decoded['model']}';
          _resultIsError = false;
        });
      } else {
        setState(() {
          _result = _extractApiError(response);
          _resultIsError = true;
        });
      }
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _result =
            'Could not reach the prediction API. Check the URL and internet connection.\n$error';
        _resultIsError = true;
      });
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('PM2.5 Air-Quality Predictor'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.all(18),
            children: [
              Text(
                'Enter the current pollutant and weather measurements.',
                style: Theme.of(context).textTheme.bodyLarge,
              ),
              const SizedBox(height: 18),
              Text('Time', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 10),
              _numberField('month', 'Month', 1, 12, integer: true),
              _numberField('hour', 'Hour', 0, 23, integer: true),
              Text('Pollutants', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 10),
              _numberField('pm10', 'PM10', 0, 1000, suffix: 'µg/m³'),
              _numberField('so2', 'SO₂', 0, 500, suffix: 'µg/m³'),
              _numberField('no2', 'NO₂', 0, 500, suffix: 'µg/m³'),
              _numberField('co', 'CO', 0, 10000, suffix: 'µg/m³'),
              _numberField('o3', 'O₃', 0, 500, suffix: 'µg/m³'),
              Text('Weather', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 10),
              _numberField('temperature', 'Temperature', -40, 50, suffix: '°C'),
              _numberField('pressure', 'Pressure', 900, 1100, suffix: 'hPa'),
              _numberField('dew_point', 'Dew point', -50, 40, suffix: '°C'),
              _numberField('rainfall', 'Rainfall', 0, 100, suffix: 'mm'),
              _numberField('wind_speed', 'Wind speed', 0, 50, suffix: 'm/s'),
              DropdownButtonFormField<String>(
                initialValue: _windDirection,
                decoration: const InputDecoration(labelText: 'Wind direction'),
                items: _directions
                    .map((value) => DropdownMenuItem(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) {
                  if (value != null) setState(() => _windDirection = value);
                },
              ),
              const SizedBox(height: 14),
              DropdownButtonFormField<String>(
                initialValue: _station,
                decoration: const InputDecoration(labelText: 'Monitoring station'),
                items: _stations
                    .map((value) => DropdownMenuItem(
                          value: value,
                          child: Text(value),
                        ))
                    .toList(),
                onChanged: (value) {
                  if (value != null) setState(() => _station = value);
                },
              ),
              const SizedBox(height: 22),
              FilledButton.icon(
                onPressed: _isLoading ? null : _predict,
                icon: const Icon(Icons.analytics_outlined),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 14),
                  child: Text('Predict'),
                ),
              ),
              if (_isLoading) ...[
                const SizedBox(height: 20),
                const Center(child: CircularProgressIndicator()),
              ],
              if (_result != null) ...[
                const SizedBox(height: 20),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(18),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Icon(
                              _resultIsError
                                  ? Icons.error_outline
                                  : Icons.check_circle_outline,
                            ),
                            const SizedBox(width: 8),
                            Text(
                              _resultIsError ? 'Prediction error' : 'Prediction',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        SelectableText(_result!),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
