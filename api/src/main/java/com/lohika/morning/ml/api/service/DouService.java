package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.MLService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class DouService {

    @Value("${dou.training.set.parquet.2.file.path}")
    private String douTrainingSetParquet2FilePath;

    @Value("${dou.training.set.parquet.3.file.path}")
    private String douTrainingSetParquet3FilePath;

    @Value("${dou.training.set.parquet.vector.file.path}")
    private String douTrainingSetParquetVectorFilePath;

    @Autowired
    private MLService mlService;

    public void useLinearRegression() {
        mlService.trainLinearRegression(douTrainingSetParquet2FilePath, 100);
        mlService.trainLinearRegression(douTrainingSetParquet3FilePath, 100);
        mlService.trainLinearRegressionUsingCrossValidator(douTrainingSetParquet2FilePath, 100);
    }

    public void useKMeans() {
        mlService.trainKMeans(douTrainingSetParquetVectorFilePath);
    }

}
