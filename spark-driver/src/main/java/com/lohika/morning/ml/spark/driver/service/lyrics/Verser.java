package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Verser extends Transformer {

    private int sentencesInVerse;

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        Column verseSplitExpression = functions
                .when(
                        functions
                                .column("rowNumber")
                                .mod(sentencesInVerse)
                                .equalTo(1),
                        1
                )
                .otherwise(0);

        Dataset<Row> sententencesPreparedForSplit = sentences.withColumn("verseStart", verseSplitExpression);

        Dataset<Row> verses = sententencesPreparedForSplit
                .withColumn("verseId",
                        functions
                                .sum("verseStart")
                                .over(
                                        Window
                                                .orderBy("rowNumber")
                                                .rowsBetween(Long.MIN_VALUE, 0)
                                )
                )
                .select("rowNumber", "verseId", "label", "stemmedSentence");

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
        Verser verser = new Verser();
        if (!extra.get(new IntParam("sentencesInVerse", "sentencesInVerse", "")).isEmpty()) {
            verser.sentencesInVerse = (Integer) (extra.get(new IntParam("sentencesInVerse", "sentencesInVerse", ""))).get();
        }

        return verser;
    }

    public int getSentencesInVerse() {
        return sentencesInVerse;
    }

    @Override
    public String uid() {
        return "Verser-" + UUID.randomUUID().toString();
    }
}
