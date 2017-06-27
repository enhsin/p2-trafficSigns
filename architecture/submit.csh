#!/bin/csh -f

foreach d1 ( 6 12 24 )
    foreach d2 ( 16 32 64 )
        if ( $d1 >= $d2 ) continue
        foreach d3 ( 120 240 480 )
            @ n = 25 * $d2
            if ( $n <= $d3 ) continue
            foreach d4 ( 80 160 240 )
                if ( $d3 <= $d4 ) continue
                echo $d1 $d2 $d3 $d4
                set out = tf_${d1}_${d2}_${d3}_${d4}.sub
                cp tf.sub $out
                perl -pi -e "s/D1/$d1/g" $out
                perl -pi -e "s/D2/$d2/g" $out
                perl -pi -e "s/D3/$d3/g" $out
                perl -pi -e "s/D4/$d4/g" $out
                qsub -q standby $out
            end
        end
    end
end

