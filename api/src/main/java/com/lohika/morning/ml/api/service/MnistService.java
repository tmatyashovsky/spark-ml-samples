package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.MLlibService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class MnistService {

    @Value("${mnist.training.set.parquet.file.path}")
    private String mnistTrainingSetParquetFilePath;

    @Value("${mnist.test.set.parquet.file.path}")
    private String mnistTestSetParquetFilePath;

    @Autowired
    private MLlibService mLlibService;

    public void useLogisticRegression() {
        mLlibService.trainLogisticRegression(mnistTrainingSetParquetFilePath, mnistTestSetParquetFilePath, 10);
    }

    public void useNaiveBayes() {
        mLlibService.trainNaiveBayes(mnistTrainingSetParquetFilePath, mnistTestSetParquetFilePath);
    }

//    private verifyModel() {
//        DataFrame catForTesting = analyticsSparkContext.getSparkSession()
//                .read()
//                .format("com.databricks.spark.csv")
//                .option("header", "false")
//                .option("inferSchema", "true")
//                .load(catForTestingFilePath);
//
//        JavaRDD<Vector> catForTestingRDD = toVector(catForTesting).toJavaRDD();
//
//        JavaRDD<Double> predictionForCat = model.predict(catForTestingRDD);
//
//        System.out.println(predictionForCat.collect());
//
//        DataFrame dogForTesting = analyticsSparkContext.getSparkSession()
//                .read()
//                .format("com.databricks.spark.csv")
//                .option("header", "false")
//                .option("inferSchema", "true")
//                .load(dogForTestingFilePath);
//
//        JavaRDD<Vector> dogForTestingRDD = toVector(dogForTesting).toJavaRDD();
//
//        JavaRDD<Double> predictionForDog = model.predict(dogForTestingRDD);
//
//        System.out.println(predictionForDog.collect());
//    }

}
