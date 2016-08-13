package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;

public class ToRegressionTrainingExample implements MapFunction<Row, Row> {

    @Override
    public Row call(final Row inputRow) {
        return RowFactory.create(((Integer)inputRow.getAs("salary")).doubleValue(),
                                 Vectors.sparse(21,
                                        new int[] {0, 1, 2 + DouConverter.transformProgrammingLanguage((String)inputRow.getAs("programming_language"))},
                                        new double[] {inputRow.getAs("experience"),
                                                      DouConverter.transformEnglishLevel(inputRow.getAs("english_level")),
                                                      1D}));
    }

}