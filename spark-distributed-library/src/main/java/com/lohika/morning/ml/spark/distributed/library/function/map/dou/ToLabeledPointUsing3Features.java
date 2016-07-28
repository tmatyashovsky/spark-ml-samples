package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;

public class ToLabeledPointUsing3Features implements MapFunction<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(final Row inputRow) {
        return new LabeledPoint(Integer.valueOf(inputRow.getInt(0)).doubleValue(),
                Vectors.sparse(21,
                        new int[] {0, 1, 2 + DouConverter.transformProgrammingLanguage(inputRow.getString(3))},
                        new double[] {inputRow.getDouble(1),
                                      DouConverter.transformEnglishLevel(inputRow.getString(2)),
                                      1D}));
    }

}