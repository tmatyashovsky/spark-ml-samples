package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class Cleanser extends Transformer {

    public String inputCol = "value";
    public String outputCol = "clean";

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
        return new Cleanser();
    }

    @Override
    public String uid() {
        return "Cleanser" + UUID.randomUUID().toString();
    }

}
