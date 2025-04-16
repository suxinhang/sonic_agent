package org.cloud.sonic.agent.tools;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.lang.reflect.Field;

public class ImageSimilarity {

    static {
        try {
            // 设置 java.library.path
            String opencvLibPath = "/path/to/opencv/native/libs"; // 请将此路径替换为你的 OpenCV native 库所在的实际路径
            System.setProperty("java.library.path", opencvLibPath);

            // 确保路径更新生效
            Field fieldSysPath = ClassLoader.class.getDeclaredField("sys_paths");
            fieldSysPath.setAccessible(true);
            fieldSysPath.set(null, null);

            // 加载 OpenCV 库
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("OpenCV library loaded successfully from " + System.getProperty("java.library.path"));
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV native library. Check java.library.path: " + System.getProperty("java.library.path"));
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Failed to set java.library.path");
            e.printStackTrace();
        }
    }

    public static double getSimilarMSSIMScore(File file1, File file2, Boolean isDelete) {
        // 剩下的代码保持不变
        Mat img1 = Imgcodecs.imread(file1.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
        Mat img2 = Imgcodecs.imread(file2.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);

        if (img1.empty() || img2.empty()) {
            throw new IllegalArgumentException("Cannot load images.");
        }

        Imgproc.resize(img2, img2, img1.size());

        Mat img1Float = new Mat();
        Mat img2Float = new Mat();
        img1.convertTo(img1Float, CvType.CV_32F);
        img2.convertTo(img2Float, CvType.CV_32F);

        Mat mu1 = new Mat();
        Mat mu2 = new Mat();
        Mat sigma1 = new Mat();
        Mat sigma2 = new Mat();
        Mat sigma12 = new Mat();

        Imgproc.GaussianBlur(img1Float, mu1, new Size(11, 11), 1.5);
        Imgproc.GaussianBlur(img2Float, mu2, new Size(11, 11), 1.5);

        Mat mu1Sq = new Mat();
        Mat mu2Sq = new Mat();
        Mat mu1Mu2 = new Mat();

        Core.multiply(mu1, mu1, mu1Sq);
        Core.multiply(mu2, mu2, mu2Sq);
        Core.multiply(mu1, mu2, mu1Mu2);

        Imgproc.GaussianBlur(img1Float.mul(img1Float), sigma1, new Size(11, 11), 1.5);
        Core.subtract(sigma1, mu1Sq, sigma1);

        Imgproc.GaussianBlur(img2Float.mul(img2Float), sigma2, new Size(11, 11), 1.5);
        Core.subtract(sigma2, mu2Sq, sigma2);

        Imgproc.GaussianBlur(img1Float.mul(img2Float), sigma12, new Size(11, 11), 1.5);
        Core.subtract(sigma12, mu1Mu2, sigma12);

        Mat t1 = new Mat();
        Mat t2 = new Mat();
        Mat t3 = new Mat();

        Core.add(mu1Mu2, Scalar.all(0.0001), t1);
        Core.add(sigma12, Scalar.all(0.0001), t2);
        Core.multiply(t1, t2, t3);

        Core.add(mu1Sq, mu2Sq, t1);
        Core.add(t1, Scalar.all(0.0001), t1);
        Core.add(sigma1, sigma2, t2);
        Core.add(t2, Scalar.all(0.0001), t2);
        Core.multiply(t1, t2, t1);

        Mat ssimMap = new Mat();
        Core.divide(t3, t1, ssimMap);

        MatOfDouble meanSSIM = new MatOfDouble();
        Core.meanStdDev(ssimMap, meanSSIM, new MatOfDouble());

        double score = meanSSIM.toArray()[0];

        if (isDelete) {
            file1.delete();
            file2.delete();
        }

        return score;
    }
}
