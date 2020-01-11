function readURL(input,name) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
      
      reader.onload = function(e) {
        document.getElementById(name).src =e.target.result;
    }
      
      reader.readAsDataURL(input.files[0]);
    }
  }

  function loading(){
    $("#loading").show();
}



function validate(){

  

    document.getElementById("btn").disabled= false;
   
  
  
};

