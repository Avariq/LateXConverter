function fileOnchange() {
    const file = $('#file')[0].files[0];

    $('#file').val('');

    const fileType = file["type"];
    const validImageTypes = ["image/jpeg", "image/png"];
    if ($.inArray(fileType, validImageTypes) < 0) {
        alert('Invalid image format selected')
        return;
    }

    const fd = new FormData();
    fd.append('file', file);

    const request = $.ajax({
        url: 'http://127.0.0.1:8000/api/internal/convert',
        type: 'post',
        async: false,
        data: fd,
        contentType: false,
        processData: false,
      });
      request.done((response) => {
        console.log(response);
        $('#latex-text-out').val(response);
      });
    
      request.fail((responce) => {
        console.error(
          `The following error occurred: ${
            responce}`,
        );
      });
}

$('#copy-results-btn').click(function (e) {
    navigator.clipboard.writeText($('#latex-text-out').val());
})
