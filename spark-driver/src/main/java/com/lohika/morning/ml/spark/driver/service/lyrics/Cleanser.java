package com.lohika.morning.ml.spark.driver.service.lyrics;

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

public class Cleanser extends Transformer implements MLWritable, MLReadable {

    private String inputCol = "value";
    private String outputCol = "clean";
    private String uid;

    public Cleanser(String uid) {
        this.uid = uid;
    }

    public Cleanser() {
        this.uid = "Cleanser" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        // Remove all punctuation symbols.
        return sentences.withColumn(outputCol,
                                    functions.regexp_replace(sentences.col(inputCol), "[^\\w\\s]", ""));
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema
                .add(DataTypes.createStructField(outputCol, DataTypes.StringType, false));
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

    @Override
    public MLReader read() {
        return new DefaultParamsReader<>();
    }

    @Override
    public Cleanser load(String path) {
        return (Cleanser) read().load(path);
    }

}
