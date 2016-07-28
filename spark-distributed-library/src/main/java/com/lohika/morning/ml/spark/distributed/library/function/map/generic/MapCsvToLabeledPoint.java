package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import java.util.stream.IntStream;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Row;

public class MapCsvToLabeledPoint implements MapFunction<Row, LabeledPoint> {

    @Override
    public LabeledPoint call(Row inputRow) {
        double[] features = new double[inputRow.size() - 1];

        IntStream.range(1, inputRow.size()).forEach(i -> features[i - 1] = inputRow.getInt(i));

        return new LabeledPoint(Integer.valueOf(inputRow.getInt(0)).doubleValue(), Vectors.dense(features));
    }

}