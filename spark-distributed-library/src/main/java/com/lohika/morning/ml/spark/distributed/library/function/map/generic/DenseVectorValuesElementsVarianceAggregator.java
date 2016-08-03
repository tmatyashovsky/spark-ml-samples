package com.lohika.morning.ml.spark.distributed.library.function.map.generic;

import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Aggregator;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;

public class DenseVectorValuesElementsVarianceAggregator extends Aggregator<Row, DoubleArrayVarianceHolder, Row> {
    private final int vectorSize;
    public DenseVectorValuesElementsVarianceAggregator(int vectorSize) {
        this.vectorSize = vectorSize;
    }

    @Override
    public DoubleArrayVarianceHolder zero() {
        // NOTE: This can be easily implemented for unknown-sized rows input, if used with List instead of arrays
        // TODO: Check difference in performance for array-based and list-based implementations.
        return new DoubleArrayVarianceHolder(vectorSize);
    }

    @Override
    public DoubleArrayVarianceHolder reduce(DoubleArrayVarianceHolder buffer, Row input) {
        // TODO: How should null-values be handled?
        if (input != null) {
            double[][] featureMinusAverages = buffer.getFeatureMinusAverages();
            int counts = buffer.getTrainingSetCounts();

            double[] averages = ((DenseVector) input.getAs("averages")).values();
            double[] features = ((DenseVector) input.getAs("features")).values();
            counts++;
            for (int j = 0; j < features.length; j++) {
                featureMinusAverages[counts][j] = features[j] - averages[j];
            }

            buffer.setTrainingSetCounts(counts);
            buffer.setLabel(input.getAs("label"));
            buffer.setAverages(averages);
        }
        return buffer;
    }

    @Override
    public DoubleArrayVarianceHolder merge(DoubleArrayVarianceHolder buffer1, DoubleArrayVarianceHolder buffer2) {
        return buffer2;
    }

    @Override
    public Row finish(DoubleArrayVarianceHolder buffer) {
        double[][] featureMinusAverages = buffer.getFeatureMinusAverages();
        int counts = buffer.getTrainingSetCounts();

        double[][] covariance = new double[vectorSize][vectorSize];
        double[] variances = new double[vectorSize * vectorSize];

        int z = 0;
        for (int i = 0; i < vectorSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                for (int k = 0; k <= counts; k++) {
                    covariance[i][j] = covariance[i][j] + featureMinusAverages[k][i] * featureMinusAverages[k][j];
                }

                variances[z++] = covariance[i][j]/ (counts);
            }
        }

        return RowFactory.create(new Object[]{buffer.getLabel(), Vectors.dense(buffer.getAverages()), Vectors.dense(variances)});
    }

    @Override
    public Encoder<DoubleArrayVarianceHolder> bufferEncoder() {
        return Encoders.bean(DoubleArrayVarianceHolder.class);
    }

    @Override
    public Encoder<Row> outputEncoder() {
        // This should NOT be a RowEncoder, as it creates additional nested structure on top of the schema provided below (which is expected)
        // TODO: Find a way to pass Array directly (with Encoders.kryo it results in binary column).
        // NOTE: In spark scala api (as always), it is as easy as "Encoder[Array[Double]] = ExpressionEncoder()"..
        // ... but in java - https://issues.apache.org/jira/browse/SPARK-13128
        return RowEncoder.apply(DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("label", DataTypes.DoubleType, true),
                DataTypes.createStructField("averages", new VectorUDT(), false, Metadata.empty()),
                DataTypes.createStructField("variances", new VectorUDT(), false, Metadata.empty())
        }));
    }
}