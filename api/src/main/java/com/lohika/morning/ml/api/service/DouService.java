package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.dou.DouMLService;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class DouService {

    @Autowired
    private DouMLService douMLService;

    @Value("${dou.training.set.csv.file.path}")
    private String douTrainingSetCsvFilePath;

    @Value("${dou.regression.model.directory.path}")
    private String douRegressionModelDirectoryPath;

    @Value("${dou.clustering.model.directory.path}")
    private String douClusteringModelDirectoryPath;

    public Map<String, Object> trainDOURegressionModel() {
        return douMLService.trainDOURegressionModel(douTrainingSetCsvFilePath, douRegressionModelDirectoryPath);
    }

    public Double predictSalary(Double experience, String englishLevel, String programmingLanguage) {
        return douMLService.predictSalary(experience, englishLevel, programmingLanguage, douRegressionModelDirectoryPath);
    }

    public Map<String, Object> clusterizeITSpecialists(Integer clusters) {
        return douMLService.clusterizeITSpecialists(douTrainingSetCsvFilePath, clusters, douClusteringModelDirectoryPath);
    }

    public Level predictLevel(Integer salary, Double experience, String englishLevel) {
        Integer prediction = douMLService.predictLevel(salary, experience, englishLevel, douClusteringModelDirectoryPath);

        switch (prediction) {
            case 0: {
                return Level.JUNIOR;
            }

            case 1: {
                return Level.SENIOR;
            }

            case 2: {
                return Level.MIDDLE;
            }

            default: {
                return Level.UNKNOWN;
            }
        }
    }
}
