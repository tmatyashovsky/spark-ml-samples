package com.lohika.morning.ml.spark.driver.service.mnist;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.mllib.MapRowToMLlibLabeledPoint;
import java.awt.image.BufferedImage;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class MnistUtilityService {

    @Autowired
    private SparkSession sparkSession;

    public void convertMnistDatasetToParquet(String mnistImageFile, String mnistLabelFile,
                                             Integer imagesCount, String mnistParquetFilePath) {
        try {
            MNISTImageFile imageFile = new MNISTImageFile(mnistImageFile, "r");
            MNISTLabelFile labelFile = new MNISTLabelFile(mnistLabelFile, "r");

            List<LabeledPoint> labeledPoints = new ArrayList<>();

            for (int k = 1; k <= imagesCount; k++) {
                imageFile.setCurr(k);
                labelFile.setCurr(k);

                int[][] imageData = imageFile.data();

                double[] pixels = new double[imageFile.cols * imageFile.rows];
                int z = 0;

                for (int i = 0; i < imageFile.rows(); i++) {
                    for (int j = 0; j < imageFile.cols(); j++) {
                        pixels[z++] = imageData[i][j];
                    }
                }

                labeledPoints.add(new LabeledPoint(labelFile.data(), Vectors.dense(pixels)));
            }

            JavaSparkContext javaSparkContext = new JavaSparkContext(sparkSession.sparkContext());
            JavaRDD<LabeledPoint> labeledPointJavaRDD = javaSparkContext.parallelize(labeledPoints);

            Dataset<Row> labeledPointsDataset = sparkSession.createDataFrame(labeledPointJavaRDD, LabeledPoint.class);

            labeledPointsDataset.write().parquet(mnistParquetFilePath);
        } catch (Exception exception) {
            throw new RuntimeException();
        }
    }

    public JavaRDD<LabeledPoint> rowToLabeledPoint(Dataset<Row> parquetRow) {
        return parquetRow.javaRDD().map(new MapRowToMLlibLabeledPoint());
    }

    private static BufferedImage getImageFromArray(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        image.setRGB(0, 0, width, height, pixels, 0, width);
        return image;
    }

    public LabeledPoint loadRandomImage(String mnistImageFile, String mnistLabelFile, String modelDirectory) {
        try {
            MNISTImageFile imageFile = new MNISTImageFile(mnistImageFile, "r");
            MNISTLabelFile labelFile = new MNISTLabelFile(mnistLabelFile, "r");

            Random random = new Random();
            int curr = random.nextInt(10000);
            imageFile.setCurr(curr);
            labelFile.setCurr(curr);

            int[][] imageData = imageFile.data();

            double[] features = new double[imageFile.cols * imageFile.rows];
            int[] pixels = new int[imageFile.cols * imageFile.rows];
            int z = 0;

            for (int i = 0; i < imageFile.rows(); i++) {
                for (int j = 0; j < imageFile.cols(); j++) {
                    features[z] = imageData[i][j];
                    pixels[z] = imageData[i][j];
                    z++;
                }
            }

            // In order to save the image.
            BufferedImage image = getImageFromArray(pixels, imageFile.cols, imageFile.rows);
            ImageIO.write(image, "jpg", Paths.get(modelDirectory).resolve("Random image - " + labelFile.data() + ".jpg").toFile());

            return new LabeledPoint(labelFile.data(), Vectors.dense(features));
        } catch (Exception exception) {
            throw new RuntimeException();
        }
    }
}
