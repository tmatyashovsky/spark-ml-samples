//package com.lohika.morning.ml.api.mechanics.deeplearning4j;
//
//import com.esotericsoftware.kryo.Kryo;
//import org.apache.spark.serializer.KryoRegistrator;
//import org.nd4j.linalg.factory.Nd4j;
//
//public class Nd4jRegistrator implements KryoRegistrator {
//    @Override
//    public void registerClasses(Kryo kryo) {
//        kryo.register(Nd4j.getBackend().getNDArrayClass(), new Nd4jSerializer());
//        kryo.register(Nd4j.getBackend().getComplexNDArrayClass(), new Nd4jSerializer());
//    }
//}
