package com.lohika.morning.ml.spark.distributed.library.function.verify;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class VerifyLogisticRegressionModel implements PairFunction<LabeledPoint, Object, Object> {

    private LogisticRegressionModel model;

    public VerifyLogisticRegressionModel(LogisticRegressionModel logisticRegressionModel) {
        this.model = logisticRegressionModel;
    }

    public Tuple2<Object, Object> call(LabeledPoint labeledPoint) {
        return new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label());
    }

}
