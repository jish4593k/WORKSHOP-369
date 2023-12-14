import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.CvType;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfFloat;
import org.opencv.core.CvType;
import org.opencv.core.MatOfRect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;

public class FaceDetectionGUI extends Application {

    private final ImageView imageView = new ImageView();
    private CascadeClassifier faceCascade;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

        VBox root = new VBox();
        root.getChildren().add(imageView);

        Scene scene = new Scene(root, 800, 600);

        primaryStage.setTitle("Face Detection");
        primaryStage.setScene(scene);
        primaryStage.show();

        startCamera();
    }

    private void startCamera() {
        new Thread(() -> {
            VideoCapture camera = new VideoCapture(0);

            if (!camera.isOpened()) {
                System.out.println("Error: Camera not opened.");
                return;
            }

            Mat frame = new Mat();

            while (true) {
                if (camera.read(frame)) {
                    detectAndDisplay(frame);
                } else {
                    System.out.println("Error: Cannot read frame.");
                    break;
                }
            }

            camera.release();
        }).start();
    }

    private void detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(frame, faces, 1.1, 5, 0, new Size(30, 30), new Size());

        List<Rect> faceList = faces.toList();

        for (Rect face : faceList) {
            Imgproc.rectangle(frame, face.tl(), face.br(), new Scalar(0, 255, 0), 2);
        }

        Image image = mat2Image(frame);
        updateImageView(image);
    }

    private Image mat2Image(Mat frame) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", frame, buffer);

        InputStream is = new ByteArrayInputStream(buffer.toArray());

        try {
            BufferedImage bufferedImage = ImageIO.read(is);
            return SwingFXUtils.toFXImage(bufferedImage, null);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void updateImageView(Image image) {
        imageView.setImage(image);
    }
}
