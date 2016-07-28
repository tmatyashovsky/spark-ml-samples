package com.lohika.morning.ml.spark.distributed.library.function.map.dou;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Row;

public class ToLabeledPointUsing2Features implements MapFunction<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        return new LabeledPoint(Integer.valueOf(inputRow.getInt(0)).doubleValue(),
                Vectors.dense(inputRow.getDouble(1),
                DouConverter.transformEnglishLevel(inputRow.getString(2))));
    }



}