package com.lohika.morning.ml.spark.driver.service.lyrics;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

public class Exploder extends Transformer {

    private String pluralCol = "filteredWords";
    private String singularCol = "filteredWord";

    @Override
    public Dataset<Row> transform(Dataset<?> sentences) {
        // Create as many rows as elements in provided column.
        Column singular = functions.explode(sentences.col(pluralCol));

        return sentences.withColumn(singularCol, singular);
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema
                .add(DataTypes.createStructField(singularCol, DataTypes.StringType, false));
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return new Exploder();
    }

    @Override
    public String uid() {
        return "Exploder";
    }

}
