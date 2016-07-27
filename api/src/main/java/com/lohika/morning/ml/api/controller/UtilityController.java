package com.lohika.morning.ml.api.controller;

import com.lohika.morning.ml.spark.driver.service.DouUtilityService;
import com.lohika.morning.ml.spark.driver.service.MLlibUtilityService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class UtilityController {

    @Autowired
    private MLlibUtilityService utilityService;

    @Autowired
    private DouUtilityService douService;

    @Value("${cat.dog.csv.file.path}")
    private String catDogCsvFilePath;

    @Value("${cat.dog.training.set.parquet.file.path}")
    private String catDogParquetFilePath;

    @Value("${mnist.csv.file.path}")
    private String mnistCsvFilePath;

    @Value("${mnist.training.set.parquet.file.path}")
    private String mnistParquetFilePath;

    @Value("${dou.training.set.parquet.2.file.path}")
    private String douParquet2FilePath;

    @Value("${dou.training.set.parquet.3.file.path}")
    private String douParquet3FilePath;

    @Value("${dou.training.set.parquet.vector.file.path}")
    private String douParquetVectorFilePath;

    @RequestMapping(value = "/convert_cats_dogs", method = RequestMethod.GET)
    ResponseEntity convertCatsAndDogs() {
        utilityService.labeledPointsToParquet(catDogCsvFilePath, catDogParquetFilePath);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/convert_mnist", method = RequestMethod.GET)
    ResponseEntity convertMnist() {
        utilityService.labeledPointsToParquet(mnistCsvFilePath, mnistParquetFilePath);
        return new ResponseEntity<>(HttpStatus.OK);
    }

    @RequestMapping(value = "/convert_dou", method = RequestMethod.GET)
    ResponseEntity convertDOU() {
        douService.convertToLabeledPointUsing2Features(douParquet2FilePath);
        douService.convertToLabeledPointUsing3Features(douParquet3FilePath);
        douService.convertToVectorUsing3Features(douParquetVectorFilePath);
        return new ResponseEntity<>(HttpStatus.OK);
    }


}
