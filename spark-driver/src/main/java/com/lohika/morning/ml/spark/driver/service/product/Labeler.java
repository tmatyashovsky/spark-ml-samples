package com.lohika.morning.ml.spark.driver.service.product;

import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class Labeler extends Transformer implements MLWritable {

    private String uid;

    public Labeler(String uid) {
        this.uid = uid;
    }

    public Labeler() {
        this.uid = "Labeler" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        return dataset.withColumn("target", functions.lit("CLASS_UNKNOWN"));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(DataTypes.createStructField("target", DataTypes.StringType, false));
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Labeler> read() {
        return new DefaultParamsReader<>();
    }
}
