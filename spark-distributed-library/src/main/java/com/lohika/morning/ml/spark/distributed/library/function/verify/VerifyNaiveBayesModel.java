package com.lohika.morning.ml.spark.distributed.library.function.verify;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class VerifyNaiveBayesModel implements PairFunction<LabeledPoint, Object, Object> {

    private NaiveBayesModel model;

    public VerifyNaiveBayesModel(NaiveBayesModel naiveBayesModel) {
        this.model = naiveBayesModel;
    }

    public Tuple2<Object, Object> call(LabeledPoint labeledPoint) {
        return new Tuple2<>(model.predict(labeledPoint.features()), labeledPoint.label());
    }

}
