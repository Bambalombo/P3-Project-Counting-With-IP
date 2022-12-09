package com.example.countingapplication;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.countingapplication.databinding.ActivityMainBinding;
import com.karumi.dexter.Dexter;
import com.karumi.dexter.PermissionToken;
import com.karumi.dexter.listener.PermissionDeniedResponse;
import com.karumi.dexter.listener.PermissionGrantedResponse;
import com.karumi.dexter.listener.PermissionRequest;
import com.karumi.dexter.listener.single.PermissionListener;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.android.Utils;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.SIFT;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.osgi.OpenCVInterface;
import org.opencv.ximgproc.Ximgproc;
import org.w3c.dom.Text;

import java.io.IOException;

import okhttp3.internal.Util;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;
    ActivityResultLauncher<String> cropImage;
    Uri slice;
    Bitmap inputBitmap;
    Bitmap sliceBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        OpenCVLoader.initDebug();

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        cropImage = registerForActivityResult(new ActivityResultContracts.GetContent(), result -> {
            Intent intent = new Intent(MainActivity.this.getApplicationContext(), CropperActivity.class);
            intent.putExtra("SendImageData", result.toString());
            startActivityForResult(intent, 100);
        });

        binding.selectImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImagePermission();
            }

            private void ImagePermission() {
                Dexter.withContext(MainActivity.this)
                        .withPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                        .withListener(new PermissionListener() {
                            @Override
                            public void onPermissionGranted(PermissionGrantedResponse permissionGrantedResponse) {
                                cropImage.launch("image/*");
                            }

                            @Override
                            public void onPermissionDenied(PermissionDeniedResponse permissionDeniedResponse) {
                                Toast.makeText(MainActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();
                            }

                            @Override
                            public void onPermissionRationaleShouldBeShown(PermissionRequest permissionRequest, PermissionToken permissionToken) {
                                permissionToken.continuePermissionRequest();
                            }
                        }).check();
            }
        });

        binding.startCountingButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                templateMatching();
            }
        });
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 100 && resultCode == 101){
            String result = data.getStringExtra("CROP");
            if (result!=null){
                slice = Uri.parse(result);
            }
            try {
                inputBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), CropperActivity.inputImage);
                sliceBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), slice);
            } catch (IOException e) {
                e.printStackTrace();
            }
            //binding.inputImageView.setImageBitmap(inputBitmap);
            binding.sliceView.setImageBitmap(sliceBitmap);
            binding.sliceTxt.setText("Find amount of this object:");
            binding.inputImageTxt.setText("In this image:");
            ((Button)findViewById(R.id.startCountingButton)).setEnabled(true);
        }
    }

    private void sift(){
        SIFT sift = SIFT.create();
        Mat mat = new Mat();
        Utils.bitmapToMat(inputBitmap, mat);
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        Imgproc.cvtColor(mat,mat, Imgproc.COLOR_RGBA2GRAY);
        sift.detect(mat,keyPoints);
        Features2d.drawKeypoints(mat, keyPoints, mat);
        Utils.matToBitmap(mat, inputBitmap);
        binding.inputImageView.setImageBitmap(inputBitmap);
    }

    private void templateMatching(){
        Mat inputMat = new Mat();
        Mat sliceMat = new Mat();
        Mat resultMat = new Mat();
        Utils.bitmapToMat(inputBitmap, inputMat);
        Utils.bitmapToMat(sliceBitmap, sliceMat);
        Imgproc.matchTemplate(inputMat,sliceMat,resultMat, Imgproc.TM_CCOEFF_NORMED);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(resultMat);
        Point matchLoc;
        matchLoc = mmr.maxLoc;
        Scalar color = new Scalar(0,255,0);
        //Imgproc.rectangle(inputMat, matchLoc, new Point(matchLoc.x + sliceMat.cols(),matchLoc.y + sliceMat.rows()), color,50);
        Utils.matToBitmap(inputMat,inputBitmap);
        binding.inputImageView.setImageBitmap(inputBitmap);
    }
}
