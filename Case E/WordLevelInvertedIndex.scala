import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

object WordLevelInvertedIndex {
  def main(args: Array[String]) = {
    val sc = SparkContext.getOrCreate()
    val fileName = "FileStore/tables/main.txt"
    val result = sc.textFile(fileName)
                   .map{ line => 
                      val array = line.split(":", 2)
                      (array(0), array(1))
                    } // Split the lines on :, such that we end up with the doc_id and the text.
                  .flatMap {
                    case (doc_id, text) =>
                      val text2 = text.toLowerCase() // Do some preprocessing.
                      text2.split("""\W+""") map {
                        word => (word, doc_id, text2.indexOfSlice(word))
                      }
                  } // flatMap split on non-word characters, rdd is now a column of words, and a column of doc_id's that contain those words. Also add location inside the document.
                  .map {
                    case (word, doc_id, pos) => ((word, doc_id, pos), 1)
                  } // Add initial count
                  .reduceByKey {
                    case (n1, n2) => n1 + n2
                  } // reduce count
                  .map {
                    case ((word, doc_id, pos), n) => (word, (doc_id,pos,n))
                  } // Turn word into primary index
                  .groupBy {
                    case (word, (doc_id,pos,n)) => word
                  } // Group by word such that we can connect all the docs that contain the word
                  .map {
                    case (word, seq) => 
                      val seq2 = seq map {
                        case (_, (doc_id, pos, n)) => (doc_id, pos, n)
                      }
                    (word, seq2.mkString(", ")) // Convert doc_id, pos, n tuple to string
                  }
                 //.saveAsTextFile("Filestore/tables/out.txt")
    val df = spark.createDataFrame(result)
    val finaldf = df.toDF("word", "doc_id, pos, count")
    finaldf.show()
    //sc.stop()
  }
}

WordLevelInvertedIndex.main(Array())