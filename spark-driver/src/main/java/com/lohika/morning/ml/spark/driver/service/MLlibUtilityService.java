package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.MapCsvToLabeledPoint;
import com.lohika.morning.ml.spark.distributed.library.function.map.generic.MapParquetToLabeledPoint;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class MLlibUtilityService {

    @Autowired
    private SparkSession sparkSession;

    public void labeledPointsToParquet(String inputFilePath, String outputFilePath) {
        Dataset<Row> inputDataAsDataFrame = sparkSession.read().csv(inputFilePath);

        Dataset<LabeledPoint> inputDataAsRDD = csvToLabeledPoint(inputDataAsDataFrame);

        labeledPointsToParquet(inputDataAsRDD, outputFilePath);
    }

    public void labeledPointsToParquet(Dataset<LabeledPoint> inputData, String outputFilePath) {
        inputData.write().parquet(outputFilePath);
    }

    public Dataset<LabeledPoint> csvToLabeledPoint(Dataset<Row> csvRow) {
        return csvRow.map(new MapCsvToLabeledPoint(), Encoders.bean(LabeledPoint.class));
    }

    public Dataset<LabeledPoint> parquetToLabeledPoint(Dataset<Row> parquetRow) {
        return parquetRow.map(new MapParquetToLabeledPoint(), Encoders.bean(LabeledPoint.class));
    }
}
