package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.generic.MapCsvToLabeledPoint;
import com.lohika.morning.ml.spark.distributed.library.function.map.generic.MapParquetToLabeledPoint;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
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

        JavaRDD<LabeledPoint> inputDataAsRDD = csvToLabeledPoint(inputDataAsDataFrame);

        labeledPointsToParquet(inputDataAsRDD, outputFilePath);
    }

    public void labeledPointsToParquet(JavaRDD<LabeledPoint> inputDataAsRDD, String outputFilePath) {
        Dataset<Row> labeledPoints = sparkSession.createDataFrame(inputDataAsRDD, LabeledPoint.class);

        labeledPoints.write().parquet(outputFilePath);
    }

    public JavaRDD<LabeledPoint> csvToLabeledPoint(Dataset<Row> csvRow) {
        return csvRow.javaRDD().map(new MapCsvToLabeledPoint());
    }

    public JavaRDD<LabeledPoint> parquetToLabeledPoint(Dataset<Row> parquetRow) {
        return parquetRow.javaRDD().map(new MapParquetToLabeledPoint());
    }
}
