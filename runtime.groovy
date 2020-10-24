def tf='/home/ec2-user/anaconda3/envs/tensorflow2_p36/lib/python3.6/site-packages/tensorflow/__init__.py'
pipeline{
    agent{
        label 'master'
    }
    // environment{
    //     PYTHONPATH=$tf:$PYTHONPATH
    // }
    
    stages{
        stage("S3 check"){
            steps{
                script{
                    git branch: 'master', credentialsId:'', url: 'https://github.com/upannala/DEPRESSED.git'
                    // sh script: "cd custom && chmod +x custom.sh && ./custom.sh ${parameter_1} ${parameter_2}"


                    def cmdpath="export PYTHONPATH=/home/ec2-user/anaconda3/envs/tensorflow2_p36/lib/python3.6/site-packages/tensorflow"
                    output = """${ sh(
					        returnStdout: true,
					        script: """ ${cmdpath}		
					        """

					        ).trim()}"""
                    
                    echo "pythonpath:: ${env.path}"
                    def cmd= "aws s3 ls s3://videosdhi --recursive | awk \'{print \$4}\'"
                    output = """${ sh(
					        returnStdout: true,
					        script: """ ${cmd}		
					        """
					        ).trim()}"""
                    
                    def keys= output.tokenize()

                    for(String key:keys){
                        def uname=key.replaceAll('.mp4','')
                        cmd1= "aws s3api get-object --bucket videosdhi --key  ${key} ${key}"
                        output = """${ sh(
					        returnStdout: true,
					        script: """ ${cmd1}		
					        """
					        ).trim()}"""
                        println output
                        sleep(60)
                        cmd3="python video.py --shape-predictor shape_predictor_68_face_landmarks.dat --video ${key}"
                        output = """${ sh(
					        returnStdout: true,
					        script: """ ${cmd3}		
					        """
					        ).trim()}"""
                        
                    }
                }
            }
        }
    }
}