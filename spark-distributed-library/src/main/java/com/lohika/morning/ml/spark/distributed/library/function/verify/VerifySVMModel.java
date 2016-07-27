package com.lohika.morning.ml.spark.distributed.library.function.verify;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class VerifySVMModel implements PairFunction<LabeledPoint, Object, Object> {

    private SVMModel model;

    public VerifySVMModel(SVMModel svmModel) {
        this.model = svmModel;
    }

    public Tuple2<Object, Object> call(LabeledPoint labeledPoint) {
        return new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label());
    }

}
