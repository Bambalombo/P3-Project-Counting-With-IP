package com.example.countingapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;

import com.example.countingapplication.databinding.ActivityMainBinding;
import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.util.UUID;

public class CropperActivity extends AppCompatActivity {

    String sourceUri, destinationUri;
    public static Uri inputImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cropper);

        Intent intent = getIntent();
        if (intent.getExtras()!=null){
            sourceUri = intent.getStringExtra("SendImageData");
            inputImage = Uri.parse(sourceUri);
        }

        destinationUri = new StringBuilder(UUID.randomUUID().toString()).append(".jpg").toString();

        UCrop.Options options = new UCrop.Options();

        UCrop.of(inputImage, Uri.fromFile(new File(getCacheDir(),destinationUri)))
                .withOptions(options)
                .withMaxResultSize(2000,2000)
                .start(CropperActivity.this);
    }
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == UCrop.REQUEST_CROP) {
            final Uri resultUri = UCrop.getOutput(data);

            Intent intent = new Intent();
            intent.putExtra("CROP",resultUri+"");
            setResult(101,intent);
            finish();

        } else if (resultCode == UCrop.RESULT_ERROR) {
            final Throwable cropError = UCrop.getError(data);
        }
    }
}