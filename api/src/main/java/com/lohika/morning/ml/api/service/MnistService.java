package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.MLlibService;
import com.lohika.morning.ml.spark.driver.service.mnist.MnistUtilityService;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class MnistService {

    @Value("${mnist.training.set.image.file.path}")
    private String mnistTrainingSetImageFilePath;

    @Value("${mnist.training.set.label.file.path}")
    private String mnistTrainingSetLabelFilePath;

    @Value("${mnist.test.set.image.file.path}")
    private String mnistTestSetImageFilePath;

    @Value("${mnist.test.set.label.file.path}")
    private String mnistTestSetLabelFilePath;

    @Value("${mnist.training.set.parquet.file.path}")
    private String mnistTrainingSetParquetFilePath;

    @Value("${mnist.test.set.parquet.file.path}")
    private String mnistTestSetParquetFilePath;

    @Value("${mnist.model.directory.path}")
    private String mnistModelDirectoryPath;

    @Value("${mnist.validation.set.directory.path}")
    private String mnistValidationSetDirectoryPath;

    @Autowired
    private MLlibService mLlibService;

    @Autowired
    private MnistUtilityService mnistUtilityService;

    public Map<String, Object> trainImages() {
        return mLlibService.trainLogisticRegression(mnistTrainingSetParquetFilePath,
                                                    mnistTestSetParquetFilePath,
                                                    10,
                                                    mnistModelDirectoryPath);
    }

    public Map<String, Object> predict() {
        LogisticRegressionModel model = mLlibService.loadLogisticRegression(mnistModelDirectoryPath);

        LabeledPoint labeledPoint = mnistUtilityService.loadRandomImage(mnistTestSetImageFilePath,
                                                                        mnistTestSetLabelFilePath,
                                                                        mnistModelDirectoryPath);

        Double prediction = model.predict(labeledPoint.features());

        Map<String, Object> predictions = new HashMap<>();
        predictions.put("Prediction", prediction);
        predictions.put("Actual", labeledPoint.label());

        System.out.println("\n------------------------------------------------");
        System.out.println(predictions);
        System.out.println("------------------------------------------------\n");

        return predictions;
    }

    public void convertMnist() {
        mnistUtilityService.convertMnistDatasetToParquet(
            mnistTrainingSetImageFilePath, mnistTrainingSetLabelFilePath, 60000, mnistTrainingSetParquetFilePath);

        mnistUtilityService.convertMnistDatasetToParquet(
            mnistTestSetImageFilePath, mnistTestSetLabelFilePath, 10000, mnistTestSetParquetFilePath);
    }
}
