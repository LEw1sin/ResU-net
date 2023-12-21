$('#picture').on('change', function(){
                var imgFiles = $(this)[0].files
                for (i=0;i<imgFiles.length;i++){
                    filePath = imgFiles[i].name
                    fileFormat = filePath.split('.')[1].toLowerCase()
                    src = window.URL.createObjectURL(imgFiles[i])
                    if( !fileFormat.match(/png|jpg|jpeg/) ) {
                        alert('上传错误,文件格式必须为：png/jpg/jpeg')
                        return
                    }
                    var preview = document.getElementById("previewImg")
                    var img = document.createElement('img')
                    img.width = 200
                    img.height = 200
                    img.src = src
                    preview.appendChild(img)
                }
            })
