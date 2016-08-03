package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Row;

/**
 * Created by tmatyashovsky on 7/28/16.
 */
public class ConvertMLVectorToMLlibVector implements Function<Row, Vector> {

    @Override
    public Vector call(Row inputRow) throws Exception {
//        List<Double> averagesDouble = inputRow.getList(2);
//        double[] averages = new double[averagesDouble.size()];
//        int i = 0;
//        for (Double d: averagesDouble) {
//            averages[i++] = d;
//        }
//        return Vectors.dense(averages);
        DenseVector mlDenseVector = (DenseVector) inputRow.getAs("features");
        return Vectors.dense(mlDenseVector.values());
    }

}
