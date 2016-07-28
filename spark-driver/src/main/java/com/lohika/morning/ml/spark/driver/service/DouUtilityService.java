package com.lohika.morning.ml.spark.driver.service;

import com.lohika.morning.ml.spark.distributed.library.function.map.dou.ToLabeledPointUsing2Features;
import com.lohika.morning.ml.spark.distributed.library.function.map.dou.ToLabeledPointUsing3Features;
import com.lohika.morning.ml.spark.distributed.library.function.map.dou.ToVectorRowUsing3Features;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class DouUtilityService {

    @Value("${dou.training.set.csv.file.path}")
    private String douTrainingSetCsvFilePath;

    @Autowired
    private SparkSession sparkSession;

    public void convertToLabeledPointUsing2Features(final String outputFilePath) {
        Dataset<Row> trainingSet = readRawTrainingSet();

        trainingSet = trainingSet.select("salary", "experience", "english_level");

        labeledPointsToParquet(toLabeledPointUsing2Features(trainingSet), outputFilePath);
    }

    public void convertToLabeledPointUsing3Features(final String outputFilePath) {
        Dataset<Row> trainingSet = readRawTrainingSet();

        trainingSet = trainingSet.select("salary", "experience", "english_level", "programming_language");

        labeledPointsToParquet(toLabeledPointUsing3Features(trainingSet), outputFilePath);
    }

    public void convertToVectorUsing3Features(final String outputFilePath) {
        Dataset<Row> trainingSet = readRawTrainingSet();

        trainingSet = trainingSet.select("salary", "experience", "english_level");

        vectorsToParquet(toVectorUsing3Features(trainingSet), outputFilePath);
    }

    private Dataset<Row> readRawTrainingSet() {
        return sparkSession
                .read()
                .option("header", "true")
                .schema(getTrainingSetStructType())
                .csv(douTrainingSetCsvFilePath);
    }

    private StructType getTrainingSetStructType() {
        return new StructType(new StructField[] {
                new StructField("id", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("city", DataTypes.StringType, true, Metadata.empty()),
                new StructField("salary", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("salary_delta", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("position", DataTypes.StringType, true, Metadata.empty()),
                new StructField("experience", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("current_job_experience", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("programming_language", DataTypes.StringType, true, Metadata.empty()),
                new StructField("specialization", DataTypes.StringType, true, Metadata.empty()),
                new StructField("age", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("sex", DataTypes.StringType, true, Metadata.empty()),
                new StructField("education", DataTypes.StringType, true, Metadata.empty()),
                new StructField("university", DataTypes.StringType, true, Metadata.empty()),
                new StructField("is_student", DataTypes.BooleanType, true, Metadata.empty()),
                new StructField("english_level", DataTypes.StringType, true, Metadata.empty()),
                new StructField("company_size", DataTypes.StringType, true, Metadata.empty()),
                new StructField("company_type", DataTypes.StringType, true, Metadata.empty()),
                new StructField("domain", DataTypes.StringType, true, Metadata.empty())
        });
    }

    private Dataset<LabeledPoint> toLabeledPointUsing2Features(Dataset<Row> training) {
        return training.map(new ToLabeledPointUsing2Features(), Encoders.bean(LabeledPoint.class));
    }

    private Dataset<LabeledPoint> toLabeledPointUsing3Features(Dataset<Row> training) {
        return training.map(new ToLabeledPointUsing3Features(), Encoders.bean(LabeledPoint.class));
    }

    private Dataset<Row> toVectorUsing3Features(Dataset<Row> training) {
        StructField[] fields = {DataTypes.createStructField("features", new VectorUDT(), false)};
        StructType schema = DataTypes.createStructType(fields);
        return training.map(new ToVectorRowUsing3Features(), RowEncoder.apply(schema));
    }

    private void labeledPointsToParquet(Dataset<LabeledPoint> labeledPoints, String outputFilePath) {
        labeledPoints.write().parquet(outputFilePath);
    }

    public void vectorsToParquet(Dataset<Row> vectors, String outputFilePath) {
        vectors.write().parquet(outputFilePath);
    }
}
