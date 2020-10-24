pipeline{
    agent{
        label 'master'
    }
    
    stages{
        stage("S3 check"){
            steps{
                script{
                    // git branch: param_map1.find{ it.key == "custom_branch" }.value, credentialsId:'', url: param_map1.find{ it.key == "custom_url" }.value
                    // sh script: "cd custom && chmod +x custom.sh && ./custom.sh ${parameter_1} ${parameter_2}"
                    def cmd= "aws s3 ls s3://videosdhi --recursive | awk \'{print $4}\'"
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
                        sleep(60000)
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