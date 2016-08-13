package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ToClusteringTrainingExample implements MapFunction<Row, Row> {

    @Override
    public Row call(Row inputRow) {
        return RowFactory.create(Vectors.dense(((Integer) inputRow.getAs("salary")).doubleValue(),
                                               inputRow.getAs("experience"),
                                               DouConverter.transformEnglishLevel(inputRow.getAs("english_level"))));
    }

}