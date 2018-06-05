/*
 * Copyright 2017 The Android Things Samples Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.androidthings.imageclassifier;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.androidthings.imageclassifier.classifier.Recognition;
import com.example.androidthings.imageclassifier.classifier.TensorFlowHelper;
import com.google.android.things.contrib.driver.bmx280.Bmx280SensorDriver;
import com.google.android.things.contrib.driver.button.ButtonInputDriver;
import com.google.android.things.contrib.driver.rainbowhat.RainbowHat;
import com.squareup.picasso.Picasso;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

public class ImageClassifierActivity extends Activity {
    private static final String TAG = "ImageClassifierActivity";

    /** Camera image capture size */
    private static final int PREVIEW_IMAGE_WIDTH = 640;
    private static final int PREVIEW_IMAGE_HEIGHT = 480;
    /** Image dimensions required by TF model */
    private static final int TF_INPUT_IMAGE_WIDTH = 224;
    private static final int TF_INPUT_IMAGE_HEIGHT = 224;
    /** Dimensions of model inputs. */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    /** TF model asset files */
    private static final String LABELS_FILE = "labels.txt";
    private static final String MODEL_FILE = "mobilenet_quant_v1_224.tflite";

    private ButtonInputDriver mButtonDriver;
    private Bmx280SensorDriver mSensorDriver;
    private boolean mProcessing;

    private ImageView mImage;
    private TextView mResultText;

    private CameraHandler mCameraHandler;
    private ImagePreprocessor mImagePreprocessor;

    private Interpreter mTensorFlowLite;
    private List<String> mLabels;

    private SensorManager mSensorManager;

    private float mPressure;
    private float mTemperature;

    /**
     * Initialize the classifier that will be used to process images.
     */
    private void initClassifier() {
        try {
            mTensorFlowLite =
                    new Interpreter(TensorFlowHelper.loadModelFile(this, MODEL_FILE));
            mLabels = TensorFlowHelper.readLabels(this, LABELS_FILE);
        } catch (IOException e) {
            Log.w(TAG, "Unable to initialize TensorFlow Lite.", e);
        }
    }

    /**
     * Clean up the resources used by the classifier.
     */
    private void destroyClassifier() {
        mTensorFlowLite.close();
    }

    /**
     * Process an image and identify what is in it. When done, the method
     * {@link #onPhotoRecognitionReady(Collection)} must be called with the results of
     * the image recognition process.
     *
     * @param image Bitmap containing the image to be classified. The image can be
     *              of any size, but preprocessing might occur to resize it to the
     *              format expected by the classification process, which can be time
     *              and power consuming.
     */
    private void doRecognize(Bitmap image) {
        // Allocate space for the inference results
        byte[][] confidencePerLabel = new byte[1][mLabels.size()];
        // Allocate buffer for image pixels.
        int[] intValues = new int[TF_INPUT_IMAGE_WIDTH * TF_INPUT_IMAGE_HEIGHT];
        ByteBuffer imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE * TF_INPUT_IMAGE_WIDTH * TF_INPUT_IMAGE_HEIGHT * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());

        // Read image data into buffer formatted for the TensorFlow model
        TensorFlowHelper.convertBitmapToByteBuffer(image, intValues, imgData);

        // Run inference on the network with the image bytes in imgData as input,
        // storing results on the confidencePerLabel array.
        mTensorFlowLite.run(imgData, confidencePerLabel);

        // Get the results with the highest confidence and map them to their labels
        Collection<Recognition> results =
                TensorFlowHelper.getBestResults(confidencePerLabel, mLabels);
        // Report the results with the highest confidence
        onPhotoRecognitionReady(results);
    }

    /**
     * Initialize the camera that will be used to capture images.
     */
    private void initCamera() {
        mImagePreprocessor = new ImagePreprocessor(
                PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT,
                TF_INPUT_IMAGE_WIDTH, TF_INPUT_IMAGE_HEIGHT);
        mCameraHandler = CameraHandler.getInstance();
        mCameraHandler.initializeCamera(this,
                PREVIEW_IMAGE_WIDTH, PREVIEW_IMAGE_HEIGHT, null,
                new ImageReader.OnImageAvailableListener() {
                    @Override
                    public void onImageAvailable(ImageReader imageReader) {
                        Bitmap bitmap = mImagePreprocessor.preprocessImage(imageReader.acquireNextImage());
                        onPhotoReady(bitmap);
                    }
                });
    }

    /**
     * Clean up resources used by the camera.
     */
    private void closeCamera() {
        mCameraHandler.shutDown();
    }

    /**
     * Load the image that will be used in the classification process.
     * When done, the method {@link #onPhotoReady(Bitmap)} must be called with the image.
     */
    private void loadPhoto() {
        mCameraHandler.takePicture();
    }



    // --------------------------------------------------------------------------------------
    // NOTE: The normal codelab flow won't require you to change anything below this line,
    // although you are encouraged to read and understand it.

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_camera);
        mImage = findViewById(R.id.imageView);
        mResultText = findViewById(R.id.resultText);

        Picasso.get().load("https://i.imgur.com/DvpvklR.png").into(mImage);
        Picasso.get().setLoggingEnabled(true);
        //mImage.setImageBitmap(getStaticBitmap());

        updateStatus(getString(R.string.initializing));

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        try {
            mSensorDriver = RainbowHat.createSensorDriver();
            mSensorDriver.registerTemperatureSensor();
            mSensorDriver.registerPressureSensor();
            mSensorManager.registerDynamicSensorCallback(mDynamicSensorCallback);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //initCamera();
        //initClassifier();
        //initButton();
        //updateStatus(getString(R.string.help_message));
    }

    private void updateDisplay() {
        updateStatus(
                String.format(
                        Locale.US,
                        "Good morning, Blackfoot.\n\n" +
                                "Onboard Temperature: %.2f °C.\n" +
                                "Barometric Pressure: %.2f hPa.\n",
                                mTemperature,
                                mPressure));
    }

    // Callback used when we register the BMP280 sensor driver with the system's SensorManager.
    private SensorManager.DynamicSensorCallback mDynamicSensorCallback
            = new SensorManager.DynamicSensorCallback() {
        @Override
        public void onDynamicSensorConnected(Sensor sensor) {
            SensorEventListener listener;
            switch (sensor.getType()) {
                case Sensor.TYPE_AMBIENT_TEMPERATURE:
                    listener = mTemperatureListener;
                    break;
                case Sensor.TYPE_PRESSURE:
                    listener = mPressureListener;
                    break;
                default:
                    Log.w(TAG, "Can't register unknown sensor type " + sensor.getType());
                    return;
            }
            mSensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL);
        }

        @Override
        public void onDynamicSensorDisconnected(Sensor sensor) {
            super.onDynamicSensorDisconnected(sensor);
            SensorEventListener listener;
            switch (sensor.getType()) {
                case Sensor.TYPE_AMBIENT_TEMPERATURE:
                    listener = mTemperatureListener;
                    break;
                case Sensor.TYPE_PRESSURE:
                    listener = mPressureListener;
                    break;
                default:
                    Log.w(TAG, "Can't unregister unknown sensor type " + sensor.getType());
                    return;
            }
            mSensorManager.unregisterListener(listener, sensor);
        }
    };

    private SensorEventListener mTemperatureListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            mTemperature = event.values[0];
            updateDisplay();
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            Log.d(TAG, "accuracy changed: " + accuracy);
        }
    };

    private SensorEventListener mPressureListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            mPressure = event.values[0];
            updateDisplay();
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            Log.d(TAG, "accuracy changed: " + accuracy);
        }
    };

    /**
     * Register a GPIO button that, when clicked, will generate the {@link KeyEvent#KEYCODE_ENTER}
     * key, to be handled by {@link #onKeyUp(int, KeyEvent)} just like any regular keyboard
     * event.
     *
     * If there's no button connected to the board, the doRecognize can still be triggered by
     * sending key events using a USB keyboard or `adb shell input keyevent 66`.
     */
    private void initButton() {
        try {
            mButtonDriver = RainbowHat.createButtonCInputDriver(KeyEvent.KEYCODE_ENTER);
            mButtonDriver.register();
        } catch (IOException e) {
            Log.w(TAG, "Cannot find button. Ignoring push button. Use a keyboard instead.", e);
        }
    }

    private Bitmap getStaticBitmap() {
        return BitmapFactory.decodeResource(this.getResources(), R.drawable.penguins);
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_ENTER) {
            if (mProcessing) {
                updateStatus("Still processing, please wait");
                return true;
            }
            updateStatus("Running photo recognition");
            mProcessing = true;
            loadPhoto();
            return true;
        }
        return super.onKeyUp(keyCode, event);
    }

    /**
     * Image capture process complete
     */
    private void onPhotoReady(Bitmap bitmap) {
        mImage.setImageBitmap(bitmap);
        doRecognize(bitmap);
    }

    /**
     * Image classification process complete
     */
    private void onPhotoRecognitionReady(Collection<Recognition> results) {
        updateStatus(formatResults(results));
        mProcessing = false;
    }

    /**
     * Format results list for display
     */
    private String formatResults(Collection<Recognition> results) {
        if (results == null || results.isEmpty()) {
            return getString(R.string.empty_result);
        } else {
            StringBuilder sb = new StringBuilder();
            Iterator<Recognition> it = results.iterator();
            int counter = 0;
            while (it.hasNext()) {
                Recognition r = it.next();
                sb.append(r.getTitle());
                counter++;
                if (counter < results.size() - 1) {
                    sb.append(", ");
                } else if (counter == results.size() - 1) {
                    sb.append(" or ");
                }
            }

            return sb.toString();
        }
    }

    /**
     * Report updates to the display and log output
     */
    private void updateStatus(String status) {
        //Log.d(TAG, status);
        mResultText.setText(status);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            destroyClassifier();
        } catch (Throwable t) {
            // close quietly
        }
        try {
            closeCamera();
        } catch (Throwable t) {
            // close quietly
        }
        try {
            if (mButtonDriver != null) mButtonDriver.close();
        } catch (Throwable t) {
            // close quietly
        }
        try {
            if (mSensorDriver != null) mSensorDriver.close();
        } catch (Throwable t) {
            // close quietly
        }
        mSensorManager.unregisterDynamicSensorCallback(mDynamicSensorCallback);
        mDynamicSensorCallback = null;
    }
}
