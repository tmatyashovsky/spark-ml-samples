package com.lohika.morning.ml.spark.driver.service.mnist;//------------------------------------------------------------
//   File: MNISTLabelFile.java
//   Written for JDK 1.1 API 
//   Author: Douglas Eck  deck@unm.edu
//------------------------------------------------------------

import java.io.*;
public class MNISTLabelFile extends RandomAccessFile {
  public int count;
  public int rows;
  public int cols;
  public int curr;
  public String fn;
  public MNISTLabelFile(String fn, String mode) throws IOException, FileNotFoundException {
    super(fn,mode);
    this.fn = fn;
    if (this.readInt()!=2049) {
      System.err.println("MNIST Label Files must have magic number of 2049.");
      System.exit(0);
    }
    curr=0;
    count=this.readInt();
    rows=this.readInt();
    cols=this.readInt();
  }

  public String toString() {
    String s = new String();
    s = s + "MNIST Label File " + fn + "\n";
    s = s + " i=" + curr() + "/" + count;
    return s;
  }

  public String status() {
    return curr() + "/" + count;
  }

  public int curr() { return curr;}

  public int data() {
    int dat=0;
    try {
      dat=readUnsignedByte();
    } catch (IOException e) { 
      System.err.println(e);
    }
    setCurr(curr);
    return dat;
  }
  
  public void next() {
    try {
      if (curr()<count) {
	skipBytes(rows*cols);
	curr++;
      }
    } catch (IOException e) { 
      System.err.println(e);
    }
  } 

  public void prev() {
    try {
      if (curr()>0) {
	seek(getFilePointer()-(rows*cols));
	curr--;
      }
    } catch (IOException e) { 
      System.err.println(e);
    }
  }
  public void setCurr(int curr) {
    try {
      if (curr>0 && curr<=count) {
	this.curr=curr;
	seek(8+curr-1);
      } else {
	System.err.println(curr + " is not in the range 0 to " + count);
	System.exit(0);
      }
    } catch (IOException e) { 
      System.err.println(e);
    } 
  }

  public String name() { return fn;}
}



