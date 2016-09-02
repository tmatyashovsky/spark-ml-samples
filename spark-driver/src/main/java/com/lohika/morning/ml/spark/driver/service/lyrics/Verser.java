package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Verser extends Transformer implements MLWritable {

    private String uid;

    public Verser(String uid) {
        this.uid = uid;
    }

    public Verser() {
        this.uid = "Verser" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        Dataset<Row> verses = sentences.withColumn(
                "verseId",
                functions.floor(functions.column("rowNumber").minus(1).divide(get(sentencesInVerse()).get())).plus(1)
        );

        verses = verses.groupBy("verseId").agg(
                functions.first("rowNumber").as("rowIdForVerse"),
                functions.first("label").as("label"),
                functions.split(functions.concat_ws(" ", functions.collect_list(functions.column("stemmedSentence"))), " ").as("verses")
        );

        return verses;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                DataTypes.createStructField("verseId", DataTypes.LongType, false),
                DataTypes.createStructField("rowIdForVerse", DataTypes.LongType, false),
                DataTypes.createStructField("label", DataTypes.DoubleType, false),
                DataTypes.createStructField("verses", DataTypes.createArrayType(DataTypes.StringType), false)
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return uid;
    }

    public IntParam sentencesInVerse() {
        return new IntParam(uid, "sentencesInVerse", "");
    }

    public Integer getSentencesInVerse() {
        return (Integer) get(sentencesInVerse()).get();
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Verser> read() {
        return new DefaultParamsReader<>();
    }

}
