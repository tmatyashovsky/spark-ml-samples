package com.lohika.morning.ml.spark.distributed.library.function.verify;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import scala.Tuple2;

public class VerifyLinearRegressionModel implements PairFunction<LabeledPoint, Double, Double> {

    private LinearRegressionModel model;

    public VerifyLinearRegressionModel(LinearRegressionModel model) {
        this.model = model;
    }

    public Tuple2<Double, Double> call(LabeledPoint labeledPoint) {
        return new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label());
    }

}
