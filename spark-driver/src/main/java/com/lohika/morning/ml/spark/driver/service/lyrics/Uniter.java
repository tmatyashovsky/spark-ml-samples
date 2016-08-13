package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class Uniter extends Transformer {

    @Override
    public Dataset<Row> transform(Dataset<?> words) {
        // Unite words into a sentence again.
        Dataset<Row> stemmedSentences = words.groupBy("rowNumber", "clean", "label")
                    .agg(functions.column("rowNumber"),
                         functions.concat_ws(" ", functions.collect_list("stemmedWord")).as("stemmedSentence"));
        stemmedSentences.cache();
        stemmedSentences.count();

        return stemmedSentences;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                DataTypes.createStructField("rowNumber", DataTypes.LongType, false),
                DataTypes.createStructField("clean", DataTypes.StringType, false),
                DataTypes.createStructField("label", DataTypes.DoubleType, false),
                DataTypes.createStructField("stemmedSentence", DataTypes.StringType, false)
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return new Uniter();
    }

    @Override
    public String uid() {
        return "Uniter-" + UUID.randomUUID().toString();
    }
}
