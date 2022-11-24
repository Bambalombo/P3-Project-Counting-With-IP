package com.example.countingapplication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

public class MainActivity extends AppCompatActivity {

    private ImageView myImage;
    private Button takePictureButton;
    private Button markButton1;
    private Button markButton2;
    private Button markButton3;
    private Button markButton4;
    private Button[] buttonArray = new Button[4];

    @SuppressLint({"ClickableViewAccessibility", "MissingInflatedId"})
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        myImage = findViewById(R.id.myImage);
        takePictureButton = findViewById(R.id.takePictureButton);
        markButton1 = findViewById(R.id.markButton1);
        markButton2 = findViewById(R.id.markButton2);
        markButton3 = findViewById(R.id.markButton3);
        markButton4 = findViewById(R.id.markButton4);

        buttonArray[0] = markButton1;
        buttonArray[1] = markButton2;
        buttonArray[2] = markButton3;
        buttonArray[3] = markButton4;

        takePictureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent openCamera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(openCamera, 0);
            }
        });

        for (Button button : buttonArray){
            button.setOnTouchListener(new View.OnTouchListener() {
                @Override
                public boolean onTouch(View view, MotionEvent event) {
                    float startX = 0, startY = 0;
                    switch (event.getActionMasked()){
                        case MotionEvent.ACTION_DOWN:
                            startX = event.getX();
                            startY = event.getY();
                            break;
                        case MotionEvent.ACTION_MOVE:
                            float endX, endY;
                            float distanceX, distanceY;

                            endX = event.getX();
                            endY = event.getY();

                            distanceX = endX-startX;
                            distanceY = endY-startY;

                            button.setX(button.getX()+distanceX);
                            button.setY(button.getY()+distanceY);

                            if (button.equals(markButton1)) {
                                markButton2.setY(markButton1.getY());
                                markButton3.setX(markButton1.getX());
                            }
                            else if (button.equals(markButton2)) {
                                markButton1.setY(markButton2.getY());
                                markButton4.setX(markButton2.getX());
                            }
                            else if (button.equals((markButton3))) {
                                markButton1.setX(markButton3.getX());
                                markButton4.setY(markButton3.getY());
                            }
                            else if (button.equals(markButton4)){
                                markButton2.setX(markButton4.getX());
                                markButton3.setY(markButton4.getY());
                            }

                            startX = endX;
                            startY = endY;
                            break;
                        case MotionEvent.ACTION_UP:
                            Rect mb1Rect = new Rect();
                            markButton1.getLocalVisibleRect(mb1Rect);
                            Log.d("Coords:", String.valueOf(mb1Rect));
                            Log.d("left         :", String.valueOf(mb1Rect.left));
                            Log.d("right        :", String.valueOf(mb1Rect.right));
                            Log.d("top          :", String.valueOf(mb1Rect.top));
                            Log.d("bottom       :", String.valueOf(mb1Rect.bottom));
                            break;
                    }
                    return true;
                }
            });
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 0){
            Bitmap image = (Bitmap) data.getExtras().get("data");
            myImage.setImageBitmap(image);
        }
    }
}
