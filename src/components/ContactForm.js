import React from 'react'
import { useRef } from 'react';
import emailjs from 'emailjs-com';

const ContactForm = () => {

  const form = useRef();

  const [formStatus, setFormStatus, inputs, setInputs] = React.useState('Send')

  const onSubmit = (e) => {
    e.preventDefault()
    setFormStatus('Submitting...', e.target.value)
    alert("an email has been sent to the server")
    const { name, email, message } = e.target.elements
    let conFom = {
      name: name.value,
      email: email.value,
      message: message.value,
    }
    emailjs.sendForm('service_rpq8oxc', 'template_s7zfhwo', form.current, 'MRYu_vT6fFmYebyUF')
      .then((result) => {
          console.log(result.text);
          alert("SUCCESS!");
      }, (error) => {
          console.log(error.text);
          alert("FAILED...", error);
      });

    console.log(conFom)
    
  }

  const handleChange = (event) => {
    const name = event.target.name;
    const value = event.target.value;
    setInputs(values => ({...values, [name]: value}))
  }

  const myStyle = {
    color: "purple",
    backgroundColor: "orange",
    padding: "10px",
    fontFamily: "Sans-Serif",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    minHeight: "200px",
    boxSizing: "border-box"
  };

  return (
    <div className="container mt-5">
      <h1 style={myStyle} className="mb-3">Contact Form Component</h1>
      <form ref={form} onSubmit={onSubmit}>
        <div className="mb-3">
          <label className="form-label" htmlFor="name">
            Name
          </label>
          <input className="form-control" type="text" name="name" id="name" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label className="form-label" htmlFor="email">
            Email
          </label>
          <input className="form-control" type="email" name="email" id="email" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label className="form-label" htmlFor="message">
            Message
          </label>
        </div>
        <div className="mb-3">  
          <textarea className="form-control" name="message" id="message" onChange={handleChange} required />
          <span>Thank you for your message !</span>
        </div>
        <button className="btn btn-danger" type="submit">
        {formStatus}
        </button>
      </form>
    </div>
  )
}

export default ContactForm