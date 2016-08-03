//package com.lohika.morning.ml.api.mechanics.deeplearning4j;
//
///**
// * Created by tmatyashovsky on 8/2/16.
// */
//public class App {
//
//    public static void load(String name){
//        try{
//            System.out.println("Trying to load: "+name);
//            System.loadLibrary(name);
//        }catch (Throwable e){
//            System.out.println("Failed: "+e.getMessage());
//            return;
//        }
//        System.out.println("Success");
//    }
//
//    public static void main(String[] args) {
//        load("libwinpthread-1");
//        load("libstdc++-6");
//        load("libquadmath-0");
//        load("libopenblas");
//        load("libgomp-1");
//        load("libgfortran-3");
//        load("libgcc_s_seh-1");
//    }
//}