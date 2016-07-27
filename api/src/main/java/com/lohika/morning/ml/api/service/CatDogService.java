package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.MLlibService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class CatDogService {

    @Value("${cat.dog.training.set.parquet.file.path}")
    private String catDogTrainingSetParquetFilePath;

    @Value("${cat.dog.test.set.parquet.file.path}")
    private String catDogTestSetParquetFilePath;

    @Autowired
    private MLlibService mLlibService;

    public void useSVM() {
        mLlibService.trainSVM(catDogTrainingSetParquetFilePath, 100);
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
